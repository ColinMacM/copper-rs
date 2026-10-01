"""The additions to real-time chunking: rolling the observation forward, noise indexed by
absolute step, best-of-K by prefix residual, exact prefix projection and the health check
(`rtc.plan_chunk`), and what each does to chunk hand-overs in simulated episodes."""
import pytest
import torch

from test_rtc import gaussian_flow, smooth_prior
from copper_policy import flow_policy as fp
from copper_policy import rtc, rtc_sim

H = 12


def gaussian_setup(d=3):
    sigma = smooth_prior(H)
    velocity = gaussian_flow(torch.zeros(H), sigma)
    y = torch.tensor([1.5, 1.6, 1.7, 1.75, 1.8, 1.8]).unsqueeze(1)
    return velocity, y, d


def test_positional_noise_is_shared_by_chunks_that_overlap_in_time():
    a = rtc.positional_noise(5, 100, 10, 3)
    b = rtc.positional_noise(5, 104, 10, 3)
    assert torch.equal(a[4:], b[:6]), "step 104 must get the same noise whichever chunk draws it"
    assert torch.equal(a, rtc.positional_noise(5, 100, 10, 3))
    assert not torch.equal(a, rtc.positional_noise(6, 100, 10, 3))
    assert not torch.equal(a[:3], a[3:6]), "different steps get different noise"
    assert a.shape == (10, 3)


def test_rolling_the_observation_forward_is_exact_on_a_ramp():
    rate = 0.01
    x = lambda t: torch.full((6,), rate * t)
    k, length = 40, 20
    a_prev = torch.stack([x(k + i) for i in range(length)])
    obs = x(k - 1)  # the arm lags the action by one step
    for d in (1, 3, 8):
        assert torch.allclose(rtc.roll_forward(obs, a_prev, d), x(k + d - 1), atol=1e-6), d
    assert torch.equal(rtc.roll_forward(obs, a_prev, 0), obs)
    assert torch.equal(rtc.roll_forward(obs, None, 5), obs)
    assert torch.allclose(rtc.roll_forward(obs, a_prev, 99), x(k + length - 1), atol=1e-6), "clamped to what exists"


def test_projection_makes_the_frozen_steps_exact_and_without_it_they_are_not():
    velocity, y, d = gaussian_setup()
    base = dict(horizon=H, steps=5)
    free, info_free = rtc.plan_chunk(velocity, torch.zeros(1), y, d, H - y.shape[0], 0, rtc.PlanConfig(**base),
                                     torch.Generator().manual_seed(1))
    proj, info_proj = rtc.plan_chunk(velocity, torch.zeros(1), y, d, H - y.shape[0], 0,
                                     rtc.PlanConfig(project=True, **base), torch.Generator().manual_seed(1))
    assert torch.equal(proj[:d], y[:d]), "the guarantee: bit-identical to the previous chunk"
    assert (free[:d] - y[:d]).abs().max() > 1e-3, "guidance alone is approximate"
    assert info_free.residual > 1e-3 and info_proj.residual == 0.0
    assert torch.equal(proj[d:], free[d:]), "the projection touches only the frozen steps"


def test_the_health_check_reports_an_unhealthy_prefix_and_projection_cures_it():
    velocity, y, d = gaussian_setup()
    tight = dict(horizon=H, health_tol=1e-6)
    _, bad = rtc.plan_chunk(velocity, torch.zeros(1), y, d, H - y.shape[0], 0, rtc.PlanConfig(**tight),
                            torch.Generator().manual_seed(2))
    _, good = rtc.plan_chunk(velocity, torch.zeros(1), y, d, H - y.shape[0], 0,
                             rtc.PlanConfig(project=True, **tight), torch.Generator().manual_seed(2))
    assert bad.guided and not bad.healthy and bad.residual > 1e-6
    assert good.healthy and good.residual == 0.0
    _, first = rtc.plan_chunk(velocity, torch.zeros(1), None, 3, 0, 0, rtc.PlanConfig(horizon=H))
    assert not first.guided and first.healthy, "nothing to be consistent with: not a guided chunk"


def test_best_of_k_returns_the_sample_with_the_smallest_prefix_residual():
    velocity, y, d = gaussian_setup()
    s = H - y.shape[0]
    cfg = rtc.PlanConfig(horizon=H, best_of=5)
    chunk, info = rtc.plan_chunk(velocity, torch.zeros(1), y, d, s, 0, cfg, torch.Generator().manual_seed(3))
    gen = torch.Generator().manual_seed(3)
    w = rtc.prefix_weights(H, d, s).unsqueeze(1)
    y_pad = torch.zeros(H, 1)
    y_pad[: y.shape[0]] = y
    scores = []
    for _ in range(5):
        a = rtc.guided_inference(velocity, torch.zeros(1), y, H, d, s, 5, 5.0, gen)
        scores.append(float(torch.linalg.norm(w * (y_pad - a))))
    assert info.tries == 5 and info.score == pytest.approx(min(scores))
    assert info.score < max(scores), "the draws differ, so choosing among them means something"


def test_rolling_forward_keeps_the_executing_chunk_for_the_delay_and_plans_the_rest():
    velocity, y, d = gaussian_setup(d=2)
    cfg = rtc.PlanConfig(horizon=H, roll_obs=True)
    chunk, info = rtc.plan_chunk(velocity, torch.zeros(1), y, d, H - y.shape[0], 7, cfg, torch.Generator().manual_seed(4))
    assert chunk.shape == (H, 1)
    assert torch.equal(chunk[:d], y[:d]), "the steps that play during the delay are the old chunk's"
    assert info.guided and info.residual == 0.0


def test_a_request_with_nothing_executing_is_sampled_freely_and_reproducibly_with_positional_noise():
    velocity, _, _ = gaussian_setup()
    cfg = rtc.PlanConfig(horizon=H, positional_noise=True, noise_seed=9)
    a, _ = rtc.plan_chunk(velocity, torch.zeros(1), None, 3, 0, 40, cfg)
    b, _ = rtc.plan_chunk(velocity, torch.zeros(1), None, 3, 0, 40, cfg)
    c, _ = rtc.plan_chunk(velocity, torch.zeros(1), None, 3, 0, 41, cfg)
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.fixture(scope="module")
def ablation(policy):
    variants = {
        "naive": dict(use_rtc=False),
        "naive+posnoise": dict(use_rtc=False, positional_noise=True),
        "rtc": dict(use_rtc=True),
        "rtc+project": dict(use_rtc=True, project=True),
        "rtc+posnoise": dict(use_rtc=True, positional_noise=True),
        "rtc+roll": dict(use_rtc=True, roll_obs=True),
        "rtc+best4": dict(use_rtc=True, best_of=4),
        "rtc+blend": dict(use_rtc=True, blend=4),
    }
    return rtc_sim.ablate(policy, 8, 40, variants)


def test_noise_indexed_by_absolute_step_smooths_hand_overs_with_and_without_rtc(ablation):
    assert ablation["naive+posnoise"]["mean"] < 0.9 * ablation["naive"]["mean"]
    assert ablation["rtc+posnoise"]["mean"] < 0.9 * ablation["rtc"]["mean"]


def test_rolling_the_observation_forward_cuts_the_largest_jumps(ablation):
    assert ablation["rtc+roll"]["max"] < 0.8 * ablation["rtc"]["max"], ablation


def test_best_of_k_smooths_hand_overs(ablation):
    assert ablation["rtc+best4"]["mean"] < 0.95 * ablation["rtc"]["mean"]


def test_the_governor_blend_smooths_without_costing_accuracy(ablation):
    blend, plain = ablation["rtc+blend"], ablation["rtc"]
    assert blend["mean"] < 0.6 * plain["mean"] and blend["max"] < 0.6 * plain["max"]
    assert blend["accel"] < 0.9 * plain["accel"], "smoother by a second measure, not only the jump"
    assert blend["final_err"] <= 1.1 * plain["final_err"] + 1e-3, "and it still ends where the plan does"


def test_projection_is_a_guarantee_not_a_speed_up(ablation):
    plain, proj = ablation["rtc"], ablation["rtc+project"]
    assert proj["unhealthy"] == 0 and plain["unhealthy"] > 0.5 * plain["guided"]
    # The steps it corrects are the ones the governor skips when the delay estimate covers the
    # delay, so what is played stays the same.
    assert proj["mean"] == pytest.approx(plain["mean"], rel=1e-6)


def test_the_plan_is_what_the_request_asks_for():
    from copper_policy import wire

    def plan(options):
        data = wire.encode_request(40, 3, 7, 2, [0.0] * 6, [0.0] * 6, options)
        return rtc.PlanConfig.from_request(wire.decode_request(data), noise_seed=9, health_tol=0.1)

    rtc_cfg = plan(wire.Options(mode=wire.MODE_RTC, denoise_steps=8, best_of=4, beta=2.5,
                                flags=wire.FLAG_PROJECT | wire.FLAG_ROLL_OBS | wire.FLAG_POSITIONAL_NOISE))
    assert (rtc_cfg.use_rtc, rtc_cfg.steps, rtc_cfg.best_of, rtc_cfg.beta) == (True, 8, 4, 2.5)
    assert (rtc_cfg.project, rtc_cfg.roll_obs, rtc_cfg.positional_noise) == (True, True, True)
    assert (rtc_cfg.noise_seed, rtc_cfg.health_tol, rtc_cfg.horizon) == (9, 0.1, 50)
    naive = plan(wire.Options(mode=wire.MODE_NAIVE, flags=wire.FLAG_POSITIONAL_NOISE))
    assert (naive.use_rtc, naive.project, naive.roll_obs, naive.positional_noise) == (False, False, False, True)
