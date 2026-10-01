"""Real-time chunking (Black, Galliker, Levine: "Real-Time Execution of Action Chunking Flow
Policies", arXiv:2506.07339), written from the paper.

A chunk is generated while the previous one executes. The first `d` actions of the new chunk
(`d` = inference delay in controller steps) will have been played by the time it arrives, so
they are frozen to the previous chunk's values, and the rest is inpainted to agree with them.
Inpainting is pseudoinverse-guided flow matching (paper Eq. 2-4) with a soft mask (Eq. 5) and a
clipped guidance weight. Algorithm 1 of the paper is `Chunker` plus `guided_inference`.

Conventions, as in the paper: flow time tau runs from 0 (noise) to 1 (data), the update is
A <- A + v / n (Eq. 1), and action `i` of the chunk answering observation `k` belongs to
control step `k + i`.
"""
import collections
import dataclasses
import math

import numpy as np
import torch


def prefix_weights(horizon, delay, exec_horizon):
    """Soft mask W of paper Eq. 5, a tensor of shape (horizon,).

    1 for the first `delay` actions (frozen), 0 for the last `exec_horizon` actions (beyond the
    end of the previous chunk), and in between an exponentially decaying weight that accounts
    for the future being more uncertain. `delay` is limited to the overlap so that it never
    freezes actions the previous chunk does not have.
    """
    h, s = int(horizon), int(exec_horizon)
    overlap = h - s
    if overlap <= 0:
        return torch.zeros(h)
    d = max(0, min(int(delay), overlap))
    w = torch.zeros(h)
    w[:d] = 1.0
    span = overlap - d + 1  # the paper's H - s - d + 1
    for i in range(d, overlap):
        c = (overlap - i) / span
        w[i] = c * (math.exp(c) - 1.0) / (math.e - 1.0)
    return w


def guidance_weight(tau, beta):
    """min(beta, (1 - tau) / (tau * r2)) of Eq. 2, with r2 from Eq. 4. Infinite at tau = 0,
    where the clip (the paper's own addition) is what keeps it finite."""
    if tau <= 0.0:
        return float(beta)
    r2 = (1.0 - tau) ** 2 / (tau**2 + (1.0 - tau) ** 2)
    return min(float(beta), (1.0 - tau) / (tau * r2))


def positional_noise(seed, start, horizon, dim):
    """Noise indexed by absolute control step: row `i` is a fixed function of `(seed, start + i)`.
    Two chunks that overlap in time therefore start their overlapping steps from the same noise,
    with nothing to remember between requests."""
    rows = [
        np.random.default_rng([int(seed) & 0xFFFFFFFF, int(t) & 0xFFFFFFFF, 7]).standard_normal(dim)
        for t in range(int(start), int(start) + int(horizon))
    ]
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def sample(velocity, obs, horizon, dim, steps, generator=None, noise=None):
    """Plain flow sampling (Eq. 1): the unguided policy, used for the first chunk."""
    a = torch.randn(horizon, dim, generator=generator) if noise is None else noise.clone()
    with torch.no_grad():
        for k in range(steps):
            a = a + velocity(a, obs, k / steps) / steps
    return a


def guided_inference(velocity, obs, a_prev, horizon, delay, exec_horizon, steps, beta, generator=None, noise=None):
    """GUIDEDINFERENCE of Algorithm 1.

    `velocity(a, obs, tau)` maps a (H, M) chunk to its (H, M) velocity and must be
    differentiable. `a_prev` is (L, M) with L <= H: the previous chunk's actions that have not
    been played yet, the first of which belongs to the control step of `obs`. It is padded on
    the right to H, where the mask is zero. `exec_horizon` is `s`, the number of steps of the
    previous chunk that were played before this inference started. Returns the new (H, M) chunk.
    """
    dim = a_prev.shape[1]
    w = prefix_weights(horizon, delay, exec_horizon).unsqueeze(1)
    y = torch.zeros(horizon, dim)
    length = min(a_prev.shape[0], horizon)
    y[:length] = a_prev[:length]
    a = torch.randn(horizon, dim, generator=generator) if noise is None else noise.clone()
    for k in range(steps):
        tau = k / steps
        a = a.detach().requires_grad_(True)
        v = velocity(a, obs, tau)
        a1_hat = a + (1.0 - tau) * v  # Eq. 3
        err = (y - a1_hat) * w  # (Y - A1)^T diag(W)
        # Vector-Jacobian product of Eq. 2, by reverse-mode autodiff.
        (g,) = torch.autograd.grad(a1_hat, a, grad_outputs=err.detach())
        step = v.detach() + guidance_weight(tau, beta) * g
        a = (a.detach() + step / steps).detach()  # Eq. 1 with the guided velocity
    return a


class Chunker:
    """The bookkeeping of Algorithm 1 outside the model: which actions of the previous chunk
    are still to be played, the delay estimate, and when to start the next inference.

    The controller here is the Copper loop. Its governor reports, every cycle, which chunk is
    executing and which step plays next (`ExecState`), so `s` and the observed delay come from
    measurement, not from counting.
    """

    def __init__(self, horizon, dim, s_min, d_init, buffer_size=10, pending_timeout=25):
        if not 1 <= s_min <= horizon - d_init:
            raise ValueError("s_min must satisfy 1 <= s_min <= H - d_init")
        self.horizon, self.dim, self.s_min = horizon, dim, s_min
        self.delays = collections.deque([int(d_init)], maxlen=buffer_size)  # the queue Q
        self.chunks = collections.OrderedDict()  # chunk_seq -> (H, M) in model space
        self.pending = None  # obs_seq of a chunk sent but not yet seen executing
        self.pending_since = 0
        self.active = None  # obs_seq of the chunk the governor executes
        self.pending_timeout = pending_timeout

    def delay(self):
        """Conservative estimate of the next delay: the maximum of the recent ones."""
        return max(self.delays)

    def observe(self, exec_state):
        """Feed one `ExecState`. Records the delay when a chunk we sent starts executing."""
        if not exec_state.chunk_active:
            self.active = None
        elif exec_state.chunk_seq != self.active and exec_state.chunk_seq in self.chunks:
            # The chunk started `next_index` steps after the observation it answers: the delay
            # that chunk actually had.
            self.active = exec_state.chunk_seq
            self.delays.append(int(exec_state.next_index))
        if self.pending is not None:
            if exec_state.chunk_active and exec_state.chunk_seq >= self.pending:
                self.pending = None
            elif exec_state.has_stamp and exec_state.stamp_seq - self.pending_since > self.pending_timeout:
                self.pending = None  # the governor dropped it; do not wait forever

    def ready(self, exec_state):
        """True when the next inference should start: nothing in flight, and either no chunk
        is executing yet or `s_min` of its steps have been played (Algorithm 1, line 13)."""
        if self.pending is not None:
            return False
        if not exec_state.chunk_active or exec_state.chunk_seq not in self.chunks:
            return True
        return exec_state.next_index >= self.s_min

    def plan(self, exec_state):
        """(a_prev, delay, s) for the inference that starts now. `a_prev` is None when there is
        nothing to stay consistent with, and the first chunk is then sampled freely."""
        if not exec_state.chunk_active or exec_state.chunk_seq not in self.chunks:
            return None, self.delay(), 0
        s = int(exec_state.next_index)
        remaining = self.chunks[exec_state.chunk_seq][s:]
        if remaining.shape[0] == 0:
            return None, self.delay(), 0
        return remaining, self.delay(), s

    def sent(self, obs_seq, chunk, stamp_seq):
        """Record a chunk that was published, answering observation `obs_seq`."""
        self.chunks[obs_seq] = chunk
        while len(self.chunks) > 8:
            self.chunks.popitem(last=False)
        self.pending = obs_seq
        self.pending_since = stamp_seq


def roll_forward(obs, a_prev, delay):
    """The state the arm is expected to be in when the new chunk starts, `delay` steps after
    `obs`: the measured state plus the motion the executing chunk is about to make. `a_prev[0]`
    is the action of the observation's own step; the state at `obs` reflects the action before
    it, which is extrapolated. Exact when the arm follows a straight ramp."""
    if a_prev is None or delay < 1:
        return obs
    last = min(int(delay), a_prev.shape[0]) - 1
    base = 2 * a_prev[0] - a_prev[1] if a_prev.shape[0] > 1 else a_prev[0]
    return obs + (a_prev[last] - base)


def prefix_residual(chunk, y, delay):
    """Largest distance between the frozen steps of `chunk` and the previous chunk's values `y`,
    the thing real-time chunking is supposed to make zero. 0 if nothing is frozen."""
    n = min(int(delay), y.shape[0], chunk.shape[0])
    if n <= 0:
        return 0.0
    return float((chunk[:n] - y[:n]).abs().max())


@dataclasses.dataclass
class PlanConfig:
    """How a chunk is planned from a request."""

    use_rtc: bool = True
    horizon: int = 50
    steps: int = 5
    beta: float = 5.0
    roll_obs: bool = False  # plan from the state after the frozen prefix; prepend the prefix
    positional_noise: bool = False  # noise indexed by absolute step, shared by overlapping chunks
    noise_seed: int = 0
    best_of: int = 1  # guided samples drawn; the one with the smallest prefix residual wins
    project: bool = False  # force the frozen steps to the previous chunk's values exactly
    health_tol: float = 0.05  # prefix residual above which the sample is reported unhealthy


    @classmethod
    def from_request(cls, request, noise_seed=0, health_tol=0.05):
        """The plan the governor asked for: its options travel in every request, so the
        configuration recorded with the graph is the one that runs. `noise_seed` and
        `health_tol` belong to the process that serves the policy."""
        from . import wire

        return cls(
            use_rtc=request.mode == wire.MODE_RTC,
            horizon=request.horizon,
            steps=request.denoise_steps,
            beta=request.beta,
            roll_obs=bool(request.flags & wire.FLAG_ROLL_OBS),
            positional_noise=bool(request.flags & wire.FLAG_POSITIONAL_NOISE),
            noise_seed=noise_seed,
            best_of=request.best_of,
            project=bool(request.flags & wire.FLAG_PROJECT),
            health_tol=health_tol,
        )


@dataclasses.dataclass
class PlanInfo:
    guided: bool = False
    residual: float = 0.0  # frozen prefix residual of the returned chunk
    score: float = 0.0  # weighted prefix residual norm of the chosen sample
    tries: int = 1
    healthy: bool = True


def plan_chunk(velocity, obs, a_prev, delay, executed, obs_seq, cfg, generator=None):
    """One chunk for one request: the single code path the policy server and the simulator share.

    `a_prev` is the previous chunk's unplayed remainder (its first step belongs to the control
    step `obs_seq`), or None when nothing is executing. Returns `(chunk, PlanInfo)`."""
    h = cfg.horizon
    dim = obs.shape[0]
    info = PlanInfo()
    if not cfg.use_rtc or a_prev is None:
        noise = positional_noise(cfg.noise_seed, obs_seq, h, dim) if cfg.positional_noise else None
        return sample(velocity, obs, h, dim, cfg.steps, generator, noise), info
    info.guided = True
    prefix = None
    obs_in, y, d_eff, s_eff, start = obs, a_prev, delay, executed, obs_seq
    if cfg.roll_obs and delay >= 1 and a_prev.shape[0] > delay:
        # The policy plans from where the arm will be when its first action runs, `delay` steps
        # on; the steps in between are the executing chunk's, kept as they are.
        prefix = a_prev[:delay]
        obs_in = roll_forward(obs, a_prev, delay)
        y, d_eff, s_eff, start = a_prev[delay:], 0, executed + delay, obs_seq + delay
    w = prefix_weights(h, d_eff, s_eff).unsqueeze(1)
    y_pad = torch.zeros(h, dim)
    y_pad[: min(y.shape[0], h)] = y[:h]
    best, best_score, tries = None, None, max(1, cfg.best_of)
    for i in range(tries):
        if i == 0 and cfg.positional_noise:
            noise = positional_noise(cfg.noise_seed, start, h, dim)
        else:
            noise = None
        a = guided_inference(velocity, obs_in, y, h, d_eff, s_eff, cfg.steps, cfg.beta, generator, noise)
        score = float(torch.linalg.norm(w * (y_pad - a)))
        if best is None or score < best_score:
            best, best_score = a, score
    chunk = best
    if cfg.project and d_eff > 0:
        n = min(d_eff, y.shape[0])
        chunk = chunk.clone()
        chunk[:n] = y[:n]
    if prefix is not None:
        chunk = torch.cat([prefix, chunk[: h - prefix.shape[0]]], dim=0)
        info.residual = 0.0  # the frozen steps are the previous chunk's by construction
    else:
        info.residual = prefix_residual(chunk, y, d_eff)
    info.score, info.tries = best_score, tries
    info.healthy = info.residual <= cfg.health_tol
    return chunk, info
