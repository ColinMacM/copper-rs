"""Serve a policy over Zenoh: newest observation in, action chunk out."""
import json
import os
import sys
import threading
import time

import zenoh

from . import wire

OBS_KEY = "vla/obs"
ACTION_KEY = "vla/action"
IMAGE_KEY = "vla/img"
EXEC_KEY = "vla/exec"
REQUEST_KEY = "vla/infer"


def session(connect_port=None, listen_port=None):
    cfg = {"mode": "peer", "scouting": {"multicast": {"enabled": False}}}
    cfg["connect"] = {"endpoints": [f"tcp/127.0.0.1:{connect_port}"] if connect_port else []}
    cfg["listen"] = {"endpoints": [f"tcp/127.0.0.1:{listen_port}"] if listen_port else []}
    return zenoh.open(zenoh.Config.from_json5(json.dumps(cfg)))


class ExecMetrics:
    """What the arm was asked to do, measured on the policy's own targets before the governor's
    limits: the jump where one chunk hands over to the next, and the largest second difference
    (acceleration) of the played targets, both in ticks."""

    def __init__(self):
        self.prev = None
        self.prev2 = None
        self.prev_chunk = None
        self.switches = 0
        self.max_jump = 0.0
        self.sum_jump = 0.0
        self.max_accel = 0.0
        self.played = 0
        self.max_delay = 0

    def update(self, exec_state, chunk_ticks):
        """`chunk_ticks`: the (H, M) ticks of the active chunk, or None if unknown."""
        if exec_state.accepted:
            self.max_delay = max(self.max_delay, exec_state.accept_skip)
        if not exec_state.played or chunk_ticks is None or exec_state.next_index >= len(chunk_ticks):
            return
        target = chunk_ticks[exec_state.next_index]
        if self.prev is not None:
            if self.prev_chunk is not None and exec_state.chunk_seq != self.prev_chunk:
                self.switches += 1
                jump = float(abs(target - self.prev).max())
                self.sum_jump += jump
                self.max_jump = max(self.max_jump, jump)
            if self.prev2 is not None:
                self.max_accel = max(self.max_accel, float(abs(target - 2 * self.prev + self.prev2).max()))
        self.prev2, self.prev, self.prev_chunk = self.prev, target, exec_state.chunk_seq
        self.played += 1

    def report(self):
        return {"switches": self.switches, "max_jump": round(self.max_jump, 1),
                "mean_jump": round(self.sum_jump / max(self.switches, 1), 1),
                "max_accel": round(self.max_accel, 1), "played": self.played, "max_delay": self.max_delay}


def serve_flow(policy, connect_port, seconds, *, cfg, delay_s=0.0, kill_after=None, log=None, seed=0):
    """Answer the governor's inference requests with a flow-matching policy.

    The server keeps no schedule. The scheduler in the governor decides when to ask and sends
    everything real-time chunking needs: the observation, the delay estimate `d`, the number of
    steps `s` of the executing chunk already played, and that chunk's unplayed remainder. With
    `cfg.use_rtc` the new chunk is inpainted against the remainder (paper Algorithm 1,
    GUIDEDINFERENCE); without it the chunk is sampled freely and replaces the old one, the
    naive asynchronous baseline. `rtc.plan_chunk` holds the rest of the options (rolling the
    observation forward, noise indexed by absolute step, best-of-K, exact prefix projection,
    the health check). A newer request replaces an unanswered older one.
    """
    import numpy as np
    import torch

    from . import flow_policy, rtc

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    sess = session(connect_port=connect_port)
    pub = sess.declare_publisher(ACTION_KEY)
    metrics = ExecMetrics()
    lock = threading.Lock()
    wake = threading.Event()
    sent_ticks = {}  # only for the metrics below; no decision reads it
    latest = {"request": None}
    stats = {"chunks": 0, "guided": 0, "events": 0, "unhealthy": 0, "bad_requests": 0, "infer_ms": [], "max_residual": 0.0}

    def on_request(sample):
        try:
            request = wire.decode_request(bytes(sample.payload))
        except ValueError:
            with lock:
                stats["bad_requests"] += 1
            return
        with lock:
            latest["request"] = request
        wake.set()

    def on_exec(sample):
        try:
            e = wire.decode_exec(bytes(sample.payload))
        except ValueError:
            return
        with lock:
            metrics.update(e, sent_ticks.get(e.chunk_seq))

    sess.declare_subscriber(REQUEST_KEY, on_request)
    sess.declare_subscriber(EXEC_KEY, on_exec)
    end = time.monotonic() + seconds
    killed_at = time.monotonic() + kill_after if kill_after else None
    next_report = time.monotonic() + 1.0
    while time.monotonic() < end:
        if log and time.monotonic() >= next_report:
            next_report += 1.0
            with lock:
                line = dict(chunks=stats["chunks"], guided=stats["guided"], events=stats["events"],
                            unhealthy=stats["unhealthy"], max_residual=round(stats["max_residual"] * flow_policy.SCALE, 1),
                            **metrics.report())
            print(json.dumps(line), file=log, flush=True)
        if killed_at and time.monotonic() >= killed_at:
            os._exit(0)
        wake.wait(0.05)
        wake.clear()
        with lock:
            request, latest["request"] = latest["request"], None
        if request is None:
            continue
        obs = torch.tensor(flow_policy.from_ticks(np.asarray(request.state[:flow_policy.DIM], dtype=np.float32)))
        t0 = time.monotonic()
        a_prev = None
        if request.previous:
            a_prev = torch.tensor(
                flow_policy.from_ticks(np.asarray(request.previous, dtype=np.float32).reshape(-1, flow_policy.DIM))
            )
        chunk, info = rtc.plan_chunk(policy.velocity, obs, a_prev, request.delay, request.executed, request.obs_seq, cfg)
        stats["guided"] += info.guided
        stats["unhealthy"] += info.guided and not info.healthy
        stats["max_residual"] = max(stats["max_residual"], info.residual)
        stats["events"] += bool(request.reason & wire.REASON_EVENT)
        stats["infer_ms"].append((time.monotonic() - t0) * 1000)
        if delay_s:
            time.sleep(delay_s)
        ticks = flow_policy.to_ticks(chunk).detach().numpy()
        with lock:
            sent_ticks[request.obs_seq] = ticks
            for old in [k for k in sent_ticks if k < request.obs_seq - 400]:
                del sent_ticks[old]
        pub.put(wire.encode_chunk(request.obs_seq, ticks.flatten().tolist()))
        stats["chunks"] += 1
    sess.close()
    stats["metrics"] = metrics.report()
    return stats


def serve(policy, connect_port, seconds, *, kill_after=None, inject=None, delay_s=0.0, log=None):
    """Runs until `seconds` elapse. `inject` (test switches): "nan" poisons every 5th chunk, "garbage" interleaves undecodable messages,
    "wild" is handled by the policy itself, "stale" holds each chunk back by `delay_s`.
    Returns a dict of counters."""
    s = session(connect_port=connect_port)
    pub = s.declare_publisher(ACTION_KEY)
    # Newest-wins slots filled by Zenoh's own threads, so the loop below sleeps on an event and
    # wakes the moment an observation lands, instead of polling on a timer. The policy always
    # works on the newest observation; an older one that was never taken is overwritten.
    latest = {"obs": None, "img": None}
    wake = threading.Event()
    lock = threading.Lock()

    def on_obs(sample):
        with lock:
            latest["obs"] = bytes(sample.payload)
        wake.set()

    def on_img(sample):
        with lock:
            latest["img"] = bytes(sample.payload)

    sub = s.declare_subscriber(OBS_KEY, on_obs)
    img_sub = s.declare_subscriber(IMAGE_KEY, on_img)
    stats = {"obs": 0, "chunks": 0, "bad_obs": 0, "infer_ms": [], "frames": 0, "bad_frames": 0, "last_frame_tov": 0}
    end = time.monotonic() + seconds
    killed_at = time.monotonic() + kill_after if kill_after else None
    last_seq = None
    next_report = time.monotonic() + 1.0
    while time.monotonic() < end:
        if log and time.monotonic() >= next_report:
            next_report += 1.0
            print(json.dumps({"obs": stats["obs"], "chunks": stats["chunks"], "frames": stats["frames"], "bad_frames": stats["bad_frames"], "last_frame_tov": stats["last_frame_tov"]}), file=log, flush=True)
        if killed_at and time.monotonic() >= killed_at:
            os._exit(0)  # the process dies without unwinding, like a crash
        # The timeout only bounds how late a progress line or a kill can be.
        wake.wait(0.05)
        wake.clear()
        with lock:
            payload, latest["obs"] = latest["obs"], None
        if payload is None:
            continue
        try:
            seq, tov_ns, state = wire.decode_obs(payload)
        except ValueError:
            stats["bad_obs"] += 1
            continue
        if seq == last_seq:
            continue
        last_seq = seq
        stats["obs"] += 1
        t0 = time.monotonic()
        steps = policy(seq, state)
        stats["infer_ms"].append((time.monotonic() - t0) * 1000)
        flat = [v for step in steps for v in step]
        if inject == "nan" and stats["obs"] % 5 == 0:
            # Early steps, where the governor would play them: a chunk is usually replaced after
            # a step or two, so a NaN deep inside it would never be executed.
            for step in range(0, 8):
                flat[step * wire.JOINTS + step % wire.JOINTS] = float("nan")
        if delay_s:
            time.sleep(delay_s)
        if inject == "garbage" and stats["obs"] % 3 == 0:
            pub.put(b"\x01\x02\x03")  # too short to be a chunk: the link must count it, not crash
            time.sleep(0.06)  # let it be taken before a real chunk replaces it in the mailbox
        pub.put(wire.encode_chunk(seq, flat))
        stats["chunks"] += 1
        # After the answer is out: checking a frame must not delay the policy.
        with lock:
            frame, latest["img"] = latest["img"], None
        if frame is not None:
            wire.check_frame(frame, stats)
    s.close()
    return stats


def main(argv=None):
    import argparse

    from . import policies

    p = argparse.ArgumentParser(description="Serve a policy to the Copper VLA loop")
    p.add_argument("--connect-port", type=int, required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--policy", choices=["scripted", "act", "flow"], default="scripted")
    p.add_argument("--checkpoint", help="flow: trained checkpoint from vla_runner.flow_policy")
    p.add_argument("--rtc", choices=["on", "off"], default="on", help="flow: real-time chunking, or the naive asynchronous baseline")
    p.add_argument("--beta", type=float, default=5.0, help="flow: guidance weight clip")
    p.add_argument("--roll-obs", action="store_true", help="flow: plan from the state after the frozen prefix")
    p.add_argument("--positional-noise", action="store_true", help="flow: noise indexed by absolute step")
    p.add_argument("--best-of", type=int, default=1, help="flow: guided samples per chunk, best prefix residual wins")
    p.add_argument("--project", action="store_true", help="flow: force the frozen steps to the previous chunk exactly")
    p.add_argument("--health-tol", type=float, default=0.05, help="flow: prefix residual (policy units) above which a chunk is unhealthy")
    p.add_argument("--denoise-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="flow: noise seed")
    p.add_argument("--calibration", help="act: LeRobot calibration JSON (joint names shoulder_pan .. gripper)")
    p.add_argument("--target", type=float, default=None, help="scripted: fixed target in raw ticks")
    p.add_argument("--amplitude", type=float, default=400.0)
    p.add_argument("--steps", type=int, default=20, help="scripted: steps per chunk (max 50)")
    p.add_argument("--kill-after", type=float, default=None)
    p.add_argument("--inject", choices=["nan", "garbage"], default=None)
    p.add_argument("--delay-s", type=float, default=0.0)
    a = p.parse_args(argv)
    if a.policy == "flow":
        from . import flow_policy

        from . import rtc

        cfg = rtc.PlanConfig(
            use_rtc=a.rtc == "on", horizon=flow_policy.HORIZON, steps=a.denoise_steps, beta=a.beta,
            roll_obs=a.roll_obs, positional_noise=a.positional_noise, noise_seed=a.seed,
            best_of=a.best_of, project=a.project, health_tol=a.health_tol,
        )
        stats = serve_flow(
            flow_policy.load(a.checkpoint), a.connect_port, a.seconds, cfg=cfg,
            delay_s=a.delay_s, kill_after=a.kill_after, log=sys.stdout, seed=a.seed,
        )
        stats["infer_ms"] = len(stats["infer_ms"])
        print(json.dumps(stats), flush=True)
        return 0
    if a.policy == "act":
        modes = {
            "shoulder_pan": "range_m100_100", "shoulder_lift": "range_m100_100", "elbow_flex": "range_m100_100",
            "wrist_flex": "range_m100_100", "wrist_roll": "range_m100_100", "gripper": "range_0_100",
        }
        policy = policies.build_act(a.calibration, modes)
    else:
        policy = policies.Scripted(amplitude=a.amplitude, target=a.target, steps=a.steps)
    stats = serve(policy, a.connect_port, a.seconds, kill_after=a.kill_after, inject=a.inject, delay_s=a.delay_s, log=sys.stdout)
    stats["infer_ms"] = len(stats["infer_ms"])
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
