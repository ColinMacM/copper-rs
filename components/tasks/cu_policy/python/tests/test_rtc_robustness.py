"""Real-time chunking when the delay differs from the constant the estimator started with.

The policy is a small network trained on synthetic demonstrations and the arm is the simulator's
stand-in that follows the played targets exactly, so these results show how the schedule
behaves, not how a real arm or a large policy would. A jump is the change of the played target
where one chunk hands over to the next; a flip is a change of side of the obstacle."""
import pytest
import torch

from copper_policy import rtc_sim
from copper_policy import flow_policy as fp

EPISODES = 40
NAIVE = dict(use_rtc=False)
RTC = dict(use_rtc=True)


def ablate(policy, delay, variants, **common):
    return rtc_sim.ablate(policy, delay, EPISODES, {k: {**common, **v} for k, v in variants.items()})


# --- the simulator's mechanics ---------------------------------------------------------------


def played_with_offsets(offsets):
    """A played path whose joint 0 sits `offsets` away from the straight path."""
    t = torch.arange(len(offsets)).float()
    straight = fp.episode_path(torch.zeros(fp.DIM), torch.zeros(1), t)
    played = straight.clone()
    played[:, 0] += torch.tensor(offsets)
    return played


def test_a_flip_is_a_change_of_side_beyond_the_threshold():
    assert rtc_sim.mode_flips(played_with_offsets([0.0, 0.3, 0.5, 0.3, 0.0, -0.3, -0.5])) == 1
    assert rtc_sim.mode_flips(played_with_offsets([0.0, 0.3, 0.5, 0.3, 0.0])) == 0
    assert rtc_sim.mode_flips(played_with_offsets([0.05, -0.05, 0.05, -0.05])) == 0, "inside the threshold"
    assert rtc_sim.mode_flips(played_with_offsets([0.3, -0.3, 0.3, -0.3])) == 3


def test_a_delay_is_a_number_a_list_or_a_seeded_function():
    import random

    rng = random.Random(0)
    assert rtc_sim._delay_of(7, 3, rng) == 7
    assert [rtc_sim._delay_of([4, 9], i, rng) for i in range(4)] == [4, 9, 4, 9]
    jitter = rtc_sim.jittered(4, 12)
    first = [jitter(i, random.Random(5)) for i in range(5)]
    assert first == [jitter(i, random.Random(5)) for i in range(5)], "the same seed gives the same delays"
    assert all(4 <= d <= 12 for d in [jitter(0, random.Random(s)) for s in range(50)])


def test_a_chunk_older_than_max_age_is_refused_and_the_arm_stays_put(policy):
    counts = {}
    played, jumps, _ = rtc_sim.run_episode(policy, use_rtc=True, delay=30, max_age=18, stats=counts)
    # Every chunk that arrived before the episode ended was refused; the last one is still in flight.
    assert counts["refused"] > 0 and counts["chunks"] - counts["refused"] <= 1
    assert jumps == [] and float(played.abs().max()) == 0.0, "nothing was ever accepted"


def test_every_chunk_is_accepted_without_a_max_age(policy):
    counts = {}
    rtc_sim.run_episode(policy, use_rtc=True, delay=30, stats=counts)
    assert counts["refused"] == 0


# --- what the schedule does under the conditions ---------------------------------------------


@pytest.fixture(scope="module")
def constant(policy):
    return ablate(policy, 8, {"naive": NAIVE, "rtc": RTC})


def test_rtc_changes_side_of_the_obstacle_less_often_than_naive_execution(constant):
    naive, rtc = constant["naive"]["flips"], constant["rtc"]["flips"]
    assert rtc < 0.85 * naive, (rtc, naive)


def test_a_varying_delay_keeps_the_mean_gain_and_widens_the_tail(policy, constant):
    jitter = ablate(policy, rtc_sim.jittered(4, 12), {"naive": NAIVE, "rtc": RTC})
    assert jitter["rtc"]["mean"] < 0.7 * jitter["naive"]["mean"]
    assert jitter["rtc"]["max"] < jitter["naive"]["max"]
    # The largest jump grows when the delay varies; it was 189 ticks at a constant delay.
    assert jitter["rtc"]["max"] > constant["rtc"]["max"]


def test_an_initial_delay_below_the_true_delay_changes_nothing(policy, constant):
    """The first measured delay replaces the initial estimate before the first guided request."""
    for d_init in (0, 3):
        r = ablate(policy, 8, {"rtc": RTC}, d_init=d_init)["rtc"]
        assert r["mean"] == pytest.approx(constant["rtc"]["mean"], rel=1e-9), d_init
        assert r["max"] == pytest.approx(constant["rtc"]["max"], rel=1e-9), d_init


def test_an_initial_delay_above_the_true_delay_stays_in_the_estimate_for_ten_chunks(policy, constant):
    high = ablate(policy, 8, {"naive": NAIVE, "rtc": RTC}, d_init=20)
    assert high["naive"]["mean"] == pytest.approx(constant["naive"]["mean"], rel=1e-9), "naive ignores d"
    assert high["rtc"]["mean"] != pytest.approx(constant["rtc"]["mean"], rel=1e-3), "the estimate changed the plan"
    assert high["rtc"]["mean"] < high["naive"]["mean"]


def test_a_delay_beyond_max_age_costs_one_chunk_and_the_gain_narrows(policy):
    spike = [6, 30, 6, 6, 6]  # the second chunk of each episode arrives after 30 cycles; max_age is 18
    hit = ablate(policy, spike, {"naive": NAIVE, "rtc": RTC}, max_age=18)
    clean = ablate(policy, [6] * 5, {"rtc": RTC}, max_age=18)
    assert hit["rtc"]["refused"] == hit["naive"]["refused"] == EPISODES, "one refused chunk per episode"
    assert hit["rtc"]["mean"] < hit["naive"]["mean"]
    assert hit["rtc"]["mean"] > clean["rtc"]["mean"], "the refused chunk leaves a worse hand-over"


@pytest.mark.parametrize("extra", [1, 2, 4])
def test_rtc_that_pays_more_delay_than_naive_still_hands_over_more_smoothly(policy, constant, extra):
    """Guided inference takes longer than free sampling, so RTC's chunks arrive later."""
    costly = ablate(policy, 8 + extra, {"rtc": RTC})["rtc"]
    naive = constant["naive"]
    assert costly["mean"] < 0.75 * naive["mean"], (extra, costly["mean"], naive["mean"])
    assert costly["max"] < naive["max"]
    assert costly["flips"] < naive["flips"]
