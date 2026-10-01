"""The real-time chunking algorithm against the paper's equations and against exact results."""
import math

import pytest
import torch

from vla_runner import rtc, wire

torch.set_num_threads(2)


def w_reference(h, d, s):
    """Paper Eq. 5, evaluated term by term."""
    out = []
    for i in range(h):
        if i < d:
            out.append(1.0)
        elif i >= h - s:
            out.append(0.0)
        else:
            c = (h - s - i) / (h - s - d + 1)
            out.append(c * (math.exp(c) - 1) / (math.e - 1))
    return out


@pytest.mark.parametrize("h,d,s", [(10, 2, 3), (50, 6, 25), (8, 0, 4), (16, 4, 4), (50, 10, 25)])
def test_prefix_weights_are_eq5(h, d, s):
    got = rtc.prefix_weights(h, d, s).tolist()
    assert got == pytest.approx(w_reference(h, d, s), abs=1e-6)


def test_prefix_weights_shape_properties():
    w = rtc.prefix_weights(50, 6, 25)
    assert (w[:6] == 1).all() and (w[25:] == 0).all()
    mid = w[6:25]
    assert ((mid > 0) & (mid < 1)).all()
    assert (mid[1:] < mid[:-1]).all(), "the decay is monotone"


def test_delay_beyond_the_overlap_freezes_only_what_exists():
    w = rtc.prefix_weights(10, delay=9, exec_horizon=5)  # the previous chunk has 5 actions left
    assert w.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert rtc.prefix_weights(10, 2, 10).tolist() == [0.0] * 10


def test_guidance_weight_is_eq2_with_the_clip():
    assert rtc.guidance_weight(0.0, 5.0) == 5.0, "infinite at tau = 0, so the clip decides"
    # The paper's Fig. 7: with n = 5 the first nonzero step, tau = 0.2, has weight 4.25.
    assert rtc.guidance_weight(0.2, 100.0) == pytest.approx(4.25)
    for tau in (0.2, 0.4, 0.6, 0.8):
        r2 = (1 - tau) ** 2 / (tau**2 + (1 - tau) ** 2)
        assert rtc.guidance_weight(tau, 1e9) == pytest.approx((1 - tau) / (tau * r2))
    assert rtc.guidance_weight(0.2, 3.0) == 3.0, "the clip binds below 4.25"


def nonlinear_velocity(a, obs, tau):
    return torch.tanh(a @ obs["m"]) - 0.5 * a + tau * obs["b"]


def test_the_autodiff_guidance_term_matches_a_finite_difference_jacobian():
    torch.manual_seed(0)
    h, m = 6, 3
    obs = {"m": torch.randn(m, m, dtype=torch.float64) * 0.7, "b": torch.randn(h, m, dtype=torch.float64)}
    a = torch.randn(h, m, dtype=torch.float64)
    y = torch.randn(h, m, dtype=torch.float64)
    w = rtc.prefix_weights(h, 2, 2).double().unsqueeze(1)
    tau = 0.4

    def a1_hat(x):
        return x + (1 - tau) * nonlinear_velocity(x, obs, tau)

    x = a.clone().requires_grad_(True)
    out = a1_hat(x)
    (g,) = torch.autograd.grad(out, x, grad_outputs=((y - out) * w).detach())

    # Finite differences of the same quantity: e^T dA1/dA with e held at its value at `a`.
    e = ((y - a1_hat(a)) * w).flatten()
    eps = 1e-6
    fd = torch.zeros(h * m, dtype=torch.float64)
    for j in range(h * m):
        bump = torch.zeros(h * m, dtype=torch.float64)
        bump[j] = eps
        plus = a1_hat(a + bump.view(h, m)).flatten()
        minus = a1_hat(a - bump.view(h, m)).flatten()
        fd[j] = (e @ (plus - minus)) / (2 * eps)
    assert torch.allclose(g.flatten(), fd, atol=1e-6)


# A flow with an exact solution: data ~ N(mu, Sigma). x_tau ~ N(tau mu, (1-tau)^2 I + tau^2 Sigma)
# and E[x1 - x0 | x_tau] = mu + (tau Sigma - (1 - tau) I) S^-1 (x - tau mu).
def gaussian_flow(mu, sigma):
    eye = torch.eye(sigma.shape[0])

    def velocity(a, obs, tau):
        s = (1 - tau) ** 2 * eye + tau**2 * sigma
        x = (a.flatten() - tau * mu).unsqueeze(1)
        v = mu.unsqueeze(1) + (tau * sigma - (1 - tau) * eye) @ torch.linalg.solve(s, x)
        return v.view_as(a)

    return velocity


def smooth_prior(h, length_scale=6.0, sd=1.0):
    t = torch.arange(h, dtype=torch.float32)
    k = sd**2 * torch.exp(-((t[:, None] - t[None, :]) ** 2) / (2 * length_scale**2))
    return k + 1e-3 * torch.eye(h)


def test_unguided_sampling_reproduces_the_exact_gaussian_flow():
    h = 12
    sigma = smooth_prior(h)
    velocity = gaussian_flow(torch.zeros(h), sigma)
    g = torch.Generator().manual_seed(1)
    xs = torch.stack([rtc.sample(velocity, None, h, 1, steps=200, generator=g).flatten() for _ in range(400)])
    cov = torch.cov(xs.T)
    assert (cov - sigma).abs().max() < 0.25, "the sampler is not reproducing the prior covariance"


def test_guidance_freezes_the_prefix_and_carries_it_into_the_continuation():
    # s = H - d makes the overlap exactly the frozen prefix: the previous chunk has d actions left.
    h, d = 12, 3
    s = h - d
    sigma = smooth_prior(h)
    velocity = gaussian_flow(torch.zeros(h), sigma)
    y = torch.tensor([1.5, 1.6, 1.7])  # the previous chunk's next actions: a smooth rise
    g = torch.Generator().manual_seed(2)
    a_prev = y.unsqueeze(1)
    guided = torch.stack([
        rtc.guided_inference(velocity, None, a_prev, h, d, s, steps=20, beta=5.0, generator=g).flatten()
        for _ in range(200)
    ])
    free = torch.stack([rtc.sample(velocity, None, h, 1, steps=20, generator=g).flatten() for _ in range(200)])

    err = lambda x: (x[:, :d] - y).abs().mean().item()
    assert err(guided) < 0.25 * err(free), (err(guided), err(free))

    # Exact Gaussian conditioning of the continuation on the frozen prefix.
    k_sp = sigma[d:, :d]
    cond_mean = k_sp @ torch.linalg.solve(sigma[:d, :d], y)
    tail = slice(d, d + 3)
    dist = lambda x: (x[:, tail].mean(0) - cond_mean[:3]).abs().mean().item()
    assert dist(guided) < 0.5 * dist(free), "the continuation does not follow the frozen prefix"
    assert guided[:, d].mean() > 0.8, "the first free action should continue the rise"


def test_the_seam_is_smoother_than_naive_switching():
    """Naive async: a fresh sample replaces the old chunk wherever it arrives. RTC: the new
    chunk's overlap agrees with the old one. The jump at the seam is the paper's failure case."""
    h, d, s = 12, 3, 4
    sigma = smooth_prior(h, length_scale=8.0)
    velocity = gaussian_flow(torch.zeros(h), sigma)
    g = torch.Generator().manual_seed(3)
    jumps_rtc, jumps_naive = [], []
    for _ in range(60):
        old = rtc.sample(velocity, None, h, 1, steps=20, generator=g)
        # The old chunk played steps 0..s; steps s.. remain, and the new one starts d steps later.
        a_prev = old[s:]
        new_rtc = rtc.guided_inference(velocity, None, a_prev, h, d, s, steps=20, beta=5.0, generator=g)
        new_naive = rtc.sample(velocity, None, h, 1, steps=20, generator=g)
        last_played = old[s + d - 1, 0]
        jumps_rtc.append((new_rtc[d, 0] - last_played).abs().item())
        jumps_naive.append((new_naive[d, 0] - last_played).abs().item())
    assert sum(jumps_rtc) < 0.4 * sum(jumps_naive), (sum(jumps_rtc), sum(jumps_naive))


def ex(stamp=0, chunk=0, index=0, flags=0):
    return wire.Exec(stamp, chunk, index, flags)


ACTIVE = wire.CHUNK_ACTIVE | wire.PLAYED


def test_chunker_delay_is_the_max_of_the_recent_observed_delays():
    c = rtc.Chunker(horizon=50, dim=2, s_min=10, d_init=3, buffer_size=3)
    assert c.delay() == 3
    for seq, delay in [(1, 2), (2, 7), (3, 4)]:
        c.sent(seq, torch.zeros(50, 2), stamp_seq=seq)
        c.observe(ex(chunk=seq, index=delay, flags=ACTIVE))
    assert c.delay() == 7
    for seq in (4, 5):
        c.sent(seq, torch.zeros(50, 2), seq)
        c.observe(ex(chunk=seq, index=1, flags=ACTIVE))
        assert c.delay() == (7 if seq == 4 else 4), "the 7 leaves the buffer of 3 after two more"


def test_chunker_plans_the_unplayed_remainder_and_waits_for_s_min():
    c = rtc.Chunker(horizon=50, dim=2, s_min=10, d_init=4)
    assert c.ready(ex())
    assert c.plan(ex()) == (None, 4, 0), "the first chunk is sampled freely"
    chunk = torch.arange(100, dtype=torch.float32).view(50, 2)
    c.sent(7, chunk, stamp_seq=7)
    assert not c.ready(ex(chunk=3, index=40, flags=ACTIVE)), "a chunk is in flight"
    c.observe(ex(chunk=7, index=4, flags=ACTIVE))  # it started executing 4 steps late
    assert not c.ready(ex(chunk=7, index=9, flags=ACTIVE)), "fewer than s_min steps played"
    state = ex(stamp=30, chunk=7, index=12, flags=ACTIVE | wire.HAS_STAMP)
    assert c.ready(state)
    remaining, d, s = c.plan(state)
    assert (d, s) == (4, 12) and remaining.shape == (38, 2)
    assert torch.equal(remaining, chunk[12:])


def test_chunker_gives_up_on_a_chunk_the_governor_never_ran():
    c = rtc.Chunker(horizon=50, dim=2, s_min=10, d_init=2, pending_timeout=25)
    c.sent(5, torch.zeros(50, 2), stamp_seq=5)
    c.observe(ex(stamp=20, chunk=1, index=30, flags=ACTIVE | wire.HAS_STAMP))
    assert c.pending == 5
    c.observe(ex(stamp=31, chunk=1, index=41, flags=ACTIVE | wire.HAS_STAMP))
    assert c.pending is None and c.ready(ex(chunk=9, index=41, flags=ACTIVE)) is True


def test_chunker_has_nothing_to_stay_consistent_with_after_the_chunk_ends():
    c = rtc.Chunker(horizon=50, dim=2, s_min=10, d_init=2)
    c.sent(1, torch.zeros(50, 2), 1)
    c.observe(ex(chunk=1, index=2, flags=ACTIVE))
    assert c.plan(ex(chunk=1, index=50, flags=wire.CHUNK_ACTIVE))[0] is None


def test_exec_state_wire_layout():
    data = wire.encode_exec(2**40 + 3, 9, 25, wire.HAS_STAMP | wire.PLAYED | wire.ACCEPTED, 7, 5, 12.5)
    assert len(data) == 36
    e = wire.decode_exec(data)
    assert (e.stamp_seq, e.chunk_seq, e.next_index) == (2**40 + 3, 9, 25)
    assert (e.accept_skip, e.reject, e.tracking_err) == (7, 5, 12.5)
    assert e.has_stamp and e.played and e.accepted and not e.chunk_active
    with pytest.raises(ValueError):
        wire.decode_exec(data[:-1])


def test_inference_request_wire_layout_and_rejections():
    data = wire.encode_request(2**40 + 9, 5, 25, wire.REASON_SCHEDULED, [1.0, 2.0, 3.0], [0.5] * 12)
    r = wire.decode_request(data)
    assert (r.obs_seq, r.delay, r.executed, r.reason) == (2**40 + 9, 5, 25, wire.REASON_SCHEDULED)
    assert r.state == [1.0, 2.0, 3.0] and r.previous == [0.5] * 12
    empty = wire.decode_request(wire.encode_request(1, 3, 0, wire.REASON_FIRST, [0.0] * 6, []))
    assert empty.previous == []
    for bad in (b"", data[:-1], data + b"\x00", data[:30]):
        with pytest.raises(ValueError):
            wire.decode_request(bad)
