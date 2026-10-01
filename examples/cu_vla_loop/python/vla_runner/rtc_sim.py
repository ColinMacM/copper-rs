"""The control loop's asynchronous schedule, simulated in Python.

It runs the same `Chunker` and the same `guided_inference` as the runner, against a stand-in for
the Copper side that follows the governor's rules: a chunk answering observation `k` that
arrives at cycle `j` starts at step `j - k`, one step plays per cycle, and an `ExecState` goes
out each cycle. The arm follows the played target exactly. This makes many episodes cheap, so the
effect of real-time chunking on chunk hand-overs can be measured with statistics; the Copper
loop itself is covered by the end-to-end test.
"""
import numpy as np
import torch

from . import flow_policy, rtc, wire


def run_episode(policy, *, use_rtc, delay, cycles=140, s_min=25, d_init=3, blend=0, seed=0, **plan):
    """One episode. `delay` is the true delay in cycles of every chunk. `blend` is the governor's
    crossfade length in steps (0 = off). `plan` holds `rtc.PlanConfig` fields (`beta`, `steps`,
    `roll_obs`, `positional_noise`, `best_of`, `project`, ...). Returns the played targets
    (cycles, M) in policy units, the hand-over jumps (max over joints, per switch) and the
    `PlanInfo` of every guided chunk."""
    dim = flow_policy.DIM
    cfg = rtc.PlanConfig(use_rtc=use_rtc, horizon=plan.pop("horizon", flow_policy.HORIZON), noise_seed=seed, **plan)
    torch.manual_seed(seed)
    chunker = rtc.Chunker(cfg.horizon, dim, s_min, d_init)
    position = torch.zeros(dim)
    active = None  # (obs_seq, chunk)
    cursor = 0
    old = None  # (chunk, cursor) the crossfade is leaving
    blended = 0
    inflight = []  # (arrival_cycle, obs_seq, chunk)
    played, jumps, infos = [], [], []
    prev_target, prev_chunk = None, None
    for t in range(cycles):
        obs = position.clone()
        for item in [i for i in inflight if i[0] == t]:
            inflight.remove(item)
            if blend and active is not None and cursor < active[1].shape[0]:
                old, blended = (active[1], cursor), 0
            active, cursor = (item[1], item[2]), t - item[1]
        flags = wire.HAS_STAMP
        index = cursor
        target = position
        if active is not None:
            flags |= wire.CHUNK_ACTIVE
            seq, chunk = active
            if cursor < chunk.shape[0]:
                target = chunk[cursor]
                if old is not None and blended < blend and old[1] < old[0].shape[0]:
                    w = (blended + 1) / (blend + 1)
                    target = (1 - w) * old[0][old[1]] + w * target
                    old = (old[0], old[1] + 1)
                    blended += 1
                else:
                    old = None
                flags |= wire.PLAYED
                cursor += 1
                if prev_chunk is not None and seq != prev_chunk and prev_target is not None:
                    jumps.append(float((target - prev_target).abs().max()))
                prev_chunk, prev_target = seq, target
        exec_state = wire.Exec(t, active[0] if active else 0, index, flags)
        position = target.clone()
        played.append(position)
        chunker.observe(exec_state)
        if chunker.ready(exec_state):
            a_prev, d, s = chunker.plan(exec_state)
            chunk, info = rtc.plan_chunk(policy.velocity, obs, a_prev, d, s, t, cfg)
            infos.append(info)
            chunker.sent(t, chunk.detach(), t)
            inflight.append((t + delay, t, chunk.detach()))
    return torch.stack(played), jumps, infos


def compare(policy, delay, episodes, **kw):
    """Mean and max hand-over jump (policy units) over `episodes` seeded episodes, RTC and naive.
    `kw` goes to `run_episode`, so a setting is applied to both."""
    out = {}
    for name, use in (("rtc", True), ("naive", False)):
        all_jumps = []
        for seed in range(episodes):
            _, jumps, _ = run_episode(policy, use_rtc=use, delay=delay, seed=seed, **kw)
            all_jumps.extend(jumps)
        out[name] = {"mean": float(np.mean(all_jumps)), "max": float(np.max(all_jumps)), "n": len(all_jumps)}
    return out


def ablate(policy, delay, episodes, variants):
    """Hand-over jumps for named variants, `{name: run_episode keyword arguments}`, over the same
    seeded episodes. Returns `{name: {"mean", "max", "n", "unhealthy", "guided", "accel", "final_err"}}`."""
    out = {}
    for name, kw in variants.items():
        jumps_all, unhealthy, guided, accels, finals = [], 0, 0, [], []
        for seed in range(episodes):
            played, jumps, infos = run_episode(policy, delay=delay, seed=seed, **kw)
            jumps_all.extend(jumps)
            accels.append(float((played[2:] - 2 * played[1:-1] + played[:-2]).abs().max()))
            finals.append(float((played[-1] - flow_policy.TARGET).abs().max()))
            unhealthy += sum(1 for i in infos if i.guided and not i.healthy)
            guided += sum(1 for i in infos if i.guided)
        out[name] = {
            "mean": float(np.mean(jumps_all)),
            "max": float(np.max(jumps_all)),
            "n": len(jumps_all),
            "unhealthy": unhealthy,
            "guided": guided,
            "accel": float(np.mean(accels)),  # largest second difference of the played targets, per episode
            "final_err": float(np.mean(finals)),  # distance from the target at the end of the episode
        }
    return out
