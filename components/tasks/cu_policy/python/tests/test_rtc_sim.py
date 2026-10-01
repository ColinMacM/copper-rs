"""Real-time chunking against naive asynchronous execution, with a trained flow policy, over many
simulated episodes (`copper_policy.rtc_sim`), plus checks that the policy itself is sound."""
import pytest
import torch

from copper_policy import flow_policy as fp
from copper_policy import rtc, rtc_sim

torch.set_num_threads(2)


def detour(chunk, t0):
    """Joint 0's offset from the straight path (no detour) for a chunk that starts at demo step
    `t0`: positive on one side of the obstacle, negative on the other."""
    steps = (t0 + torch.arange(chunk.shape[0])).float()
    straight = fp.episode_path(torch.zeros(6), torch.zeros(1), steps)
    return chunk[:, 0] - straight[:, 0]


def test_the_demo_policy_produces_smooth_chunks_on_both_sides_of_the_fork(policy):
    torch.manual_seed(0)
    steps, sides = [], []
    for _ in range(40):
        a = rtc.sample(policy.velocity, torch.zeros(6), 50, 6, 5)
        steps.append(float((a[1:] - a[:-1]).abs().max()) * fp.SCALE)
        sides.append(float(detour(a, 0)[-8:].mean()) > 0)
    assert sorted(steps)[20] < 60, f"median roughest step {sorted(steps)[20]:.0f} ticks: not a smooth trajectory"
    assert 8 <= sum(sides) <= 32, f"from the pre-fork state the policy must go both ways: {sum(sides)} of 40"


def test_after_the_fork_the_state_decides_the_side(policy):
    t_obs = fp.FORK + 14
    for mode in (1.0, -1.0):
        obs = fp.episode_path(torch.zeros(6), torch.tensor([mode]), torch.tensor([float(t_obs)]))[0]
        torch.manual_seed(1)
        kept = sum(
            1
            for _ in range(30)
            if float(detour(rtc.sample(policy.velocity, obs, 50, 6, 5), t_obs)[5:25].mean()) * mode > 0
        )
        assert kept >= 27, f"mode {mode}: only {kept} of 30 samples kept the side the state shows"


def test_rtc_hands_chunks_over_more_smoothly_than_naive_async(policy):
    for delay, mean_ratio in ((8, 0.8), (12, 0.75)):
        r = rtc_sim.compare(policy, delay, episodes=40)
        assert r["rtc"]["n"] > 100 and r["naive"]["n"] > 100
        assert r["rtc"]["mean"] < mean_ratio * r["naive"]["mean"], (delay, r)


def test_the_largest_hand_over_jump_under_long_delay_is_smaller_with_rtc(policy):
    r = rtc_sim.compare(policy, 12, episodes=40)
    assert r["rtc"]["max"] < 0.8 * r["naive"]["max"], r


def test_without_guidance_rtc_is_exactly_the_naive_baseline(policy):
    """Control: with the guidance weight clipped to zero the guided sampler draws the same noise
    and follows the unguided flow, so the schedule alone changes nothing."""
    r = rtc_sim.compare(policy, 8, episodes=10, beta=0.0)
    assert r["rtc"]["mean"] == pytest.approx(r["naive"]["mean"], rel=1e-5)
    assert r["rtc"]["max"] == pytest.approx(r["naive"]["max"], rel=1e-5)
