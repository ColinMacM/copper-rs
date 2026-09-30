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


def session(connect_port=None, listen_port=None):
    cfg = {"mode": "peer", "scouting": {"multicast": {"enabled": False}}}
    cfg["connect"] = {"endpoints": [f"tcp/127.0.0.1:{connect_port}"] if connect_port else []}
    cfg["listen"] = {"endpoints": [f"tcp/127.0.0.1:{listen_port}"] if listen_port else []}
    return zenoh.open(zenoh.Config.from_json5(json.dumps(cfg)))


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
    p.add_argument("--policy", choices=["scripted", "act"], default="scripted")
    p.add_argument("--calibration", help="act: LeRobot calibration JSON (joint names shoulder_pan .. gripper)")
    p.add_argument("--target", type=float, default=None, help="scripted: fixed target in raw ticks")
    p.add_argument("--amplitude", type=float, default=400.0)
    p.add_argument("--steps", type=int, default=20, help="scripted: steps per chunk (max 50)")
    p.add_argument("--kill-after", type=float, default=None)
    p.add_argument("--inject", choices=["nan", "garbage"], default=None)
    p.add_argument("--delay-s", type=float, default=0.0)
    a = p.parse_args(argv)
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
