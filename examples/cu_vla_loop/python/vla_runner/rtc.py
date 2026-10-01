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
import math

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


def sample(velocity, obs, horizon, dim, steps, generator=None):
    """Plain flow sampling (Eq. 1): the unguided policy, used for the first chunk."""
    a = torch.randn(horizon, dim, generator=generator)
    with torch.no_grad():
        for k in range(steps):
            a = a + velocity(a, obs, k / steps) / steps
    return a


def guided_inference(velocity, obs, a_prev, horizon, delay, exec_horizon, steps, beta, generator=None):
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
    a = torch.randn(horizon, dim, generator=generator)
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
    """The bookkeeping of Algorithm 1 that is not the model: which actions of the previous chunk
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
