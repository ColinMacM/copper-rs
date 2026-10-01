"""A small flow-matching action-chunking policy, trained here on synthetic demonstrations.

It exists so real-time chunking has a flow policy to run against without downloading a model.
The demonstrations fork on purpose. Every demonstration moves along the same path until step
FORK, then goes around an obstacle on the left or on the right. Before the fork the state does
not reveal which. That is the situation in which naive asynchronous execution fails: a chunk
computed from an observation taken just before the fork can pick one side while the previous
chunk, already executing, has started the other, and the new chunk takes over `d` steps later.
Inpainting against the previous chunk keeps the choice.

Units: the policy works on normalized joints, `(ticks - CENTER) / SCALE`; `to_ticks` and
`from_ticks` convert. It observes the six joint positions only.
"""
import math
import sys

import torch
from torch import nn

HORIZON = 50
DIM = 6
CENTER = 2048.0
# One unit of the policy's action space is SCALE ticks, so the demonstrations have a spread of
# order one, like the flow's noise; at a much smaller spread the network's error swamps them.
SCALE = 600.0
EPISODE = 110  # steps of one demonstration
FORK = 40  # step at which the detour begins
AMPLITUDE = 500.0 / SCALE  # sideways detour of joint 0
TARGET = torch.tensor([850.0, -650.0, 550.0, -450.0, 350.0, 0.0]) / SCALE
START_SPREAD = 300.0 / SCALE


def to_ticks(a):
    return a * SCALE + CENTER


def from_ticks(ticks):
    return (ticks - CENTER) / SCALE


def episode_path(start, mode, t):
    """State at step(s) `t` of a demonstration from `start` (.., DIM) with detour side `mode`
    (+1 or -1): a smooth move to TARGET with a sideways arc on joint 0. Time saturates, so the
    arm holds the target after the episode ends."""
    u = (t.clamp(max=EPISODE - 1) / (EPISODE - 1)).unsqueeze(-1)
    smooth = u * u * (3 - 2 * u)
    path = start + (TARGET - start) * smooth
    detour = ((t - FORK).clamp(min=0, max=EPISODE - 1 - FORK) / (EPISODE - 1 - FORK)).unsqueeze(-1)
    arc = mode.unsqueeze(-1) * AMPLITUDE * torch.sin(math.pi * detour)
    bump = torch.zeros_like(path)
    bump[..., 0] = arc[..., 0]
    return path + bump


def demo_batch(n, generator=None):
    """`n` (observation, chunk) pairs: a random window of a random demonstration, observed at
    the window's first state."""
    start = (torch.rand(n, DIM, generator=generator) - 0.5) * START_SPREAD
    mode = torch.where(torch.rand(n, generator=generator) < 0.5, -1.0, 1.0)
    t0 = torch.randint(0, EPISODE, (n,), generator=generator)
    steps = t0.unsqueeze(1) + torch.arange(HORIZON)
    chunk = episode_path(start.unsqueeze(1), mode.unsqueeze(1), steps.float())
    obs = chunk[:, 0]
    return obs, chunk


def dct_basis(horizon, k):
    """First `k` columns of the orthonormal DCT-II basis over time, shape (horizon, k)."""
    t = torch.arange(horizon, dtype=torch.float32)
    cols = []
    for f in range(k):
        col = torch.cos(math.pi * (t + 0.5) * f / horizon)
        cols.append(col / col.norm())
    return torch.stack(cols, dim=1)


BASIS_SIZE = 12
BASIS = dct_basis(HORIZON, BASIS_SIZE)


class FlowNet(nn.Module):
    """Velocity of the low-frequency part of a chunk: the first BASIS_SIZE DCT coefficients of
    each joint's trajectory. The demonstrations are smooth, so the rest of the chunk is zero in
    the data; its exact flow velocity is `-a / (1 - tau)` and `FlowPolicy.velocity` adds it."""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BASIS_SIZE * DIM + DIM + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, BASIS_SIZE * DIM),
        )

    def forward(self, c, obs, tau):
        """c: (B, K, M) coefficients, obs: (B, M), tau: (B,) -> velocity (B, K, M)."""
        x = torch.cat([c.flatten(1), obs, tau.unsqueeze(1)], dim=1)
        return self.net(x).view(-1, BASIS_SIZE, DIM)


def train(steps=3000, batch=512, seed=0, lr=2e-3, log=None):
    """Conditional flow matching on the coefficients: c_tau = (1 - tau) c_0 + tau c_1, target
    velocity c_1 - c_0, with c_0 the projection of N(0, I) noise (orthonormal, so still N(0, I))."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    net = FlowNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss = torch.tensor(0.0)
    for i in range(steps):
        obs, a1 = demo_batch(batch, g)
        c1 = torch.einsum("hk,bhm->bkm", BASIS, a1)
        c0 = torch.randn(c1.shape, generator=g)
        tau = torch.rand(batch, generator=g)
        c_tau = (1 - tau.view(-1, 1, 1)) * c0 + tau.view(-1, 1, 1) * c1
        loss = ((net(c_tau, obs, tau) - (c1 - c0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and i % 500 == 0:
            print(f"step {i} loss {loss.item():.4f}", file=log, flush=True)
    net.eval()
    return net


class FlowPolicy:
    """Adapts a trained `FlowNet` to the `velocity(a, obs, tau)` of `rtc`, for one observation."""

    def __init__(self, net):
        self.net = net

    def velocity(self, a, obs, tau):
        """Velocity of a full (H, M) chunk: the network's for the smooth part, the exact one
        for the remainder (the data has none, so it decays toward zero)."""
        coeffs = BASIS.T @ a
        t = torch.full((1,), float(tau))
        v_low = BASIS @ self.net(coeffs.unsqueeze(0), obs.unsqueeze(0), t)[0]
        rest = a - BASIS @ coeffs
        return v_low - rest / max(1.0 - float(tau), 1e-3)


def load(path):
    net = FlowNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return FlowPolicy(net)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Train the demo flow policy")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.set_num_threads(2)
    net = train(a.steps, seed=a.seed, log=sys.stderr)
    torch.save(net.state_dict(), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
