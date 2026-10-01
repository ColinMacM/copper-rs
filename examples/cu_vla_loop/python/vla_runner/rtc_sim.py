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


def run_episode(policy, *, use_rtc, delay, cycles=140, horizon=flow_policy.HORIZON, s_min=25,
                d_init=3, beta=5.0, steps=5, seed=0):
    """One episode. `delay` is the true delay in cycles of every chunk. Returns the played
    targets (cycles, M) in policy units and the hand-over jumps (max over joints, per switch)."""
    dim = flow_policy.DIM
    torch.manual_seed(seed)
    chunker = rtc.Chunker(horizon, dim, s_min, d_init)
    position = torch.zeros(dim)
    active = None  # (obs_seq, chunk)
    cursor = 0
    inflight = []  # (arrival_cycle, obs_seq, chunk)
    played, jumps = [], []
    prev_target, prev_chunk = None, None
    for t in range(cycles):
        obs = position.clone()
        for item in [i for i in inflight if i[0] == t]:
            inflight.remove(item)
            active, cursor = (item[1], item[2]), t - item[1]
        flags = wire.HAS_STAMP
        index = cursor
        target = position
        if active is not None:
            flags |= wire.CHUNK_ACTIVE
            seq, chunk = active
            if cursor < chunk.shape[0]:
                target = chunk[cursor]
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
            if use_rtc and a_prev is not None:
                chunk = rtc.guided_inference(policy.velocity, obs, a_prev, horizon, d, s, steps, beta)
            else:
                chunk = rtc.sample(policy.velocity, obs, horizon, dim, steps)
            chunker.sent(t, chunk.detach(), t)
            inflight.append((t + delay, t, chunk.detach()))
    return torch.stack(played), jumps


def compare(policy, delay, episodes, **kw):
    """Mean and max hand-over jump (policy units) over `episodes` seeded episodes, RTC and naive."""
    out = {}
    for name, use in (("rtc", True), ("naive", False)):
        all_jumps = []
        for seed in range(episodes):
            _, jumps = run_episode(policy, use_rtc=use, delay=delay, seed=seed, **kw)
            all_jumps.extend(jumps)
        out[name] = {"mean": float(np.mean(all_jumps)), "max": float(np.max(all_jumps)), "n": len(all_jumps)}
    return out
