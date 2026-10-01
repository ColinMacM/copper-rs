"""`copper_policy.server`: the answer to one request, and the loop of `serve`."""
import pytest

from copper_policy import server, wire


def request_bytes(seq=41, previous=()):
    return wire.encode_request(seq, 3, 7, 2, [1.0] * 6, list(previous))


def test_a_reply_names_the_observation_and_carries_the_plan():
    seen = []

    def policy(request):
        seen.append(request)
        return [request.state[0] + i for i in range(12)]

    chunk_seq, values = wire.decode_chunk(server.answer(policy, request_bytes()))
    assert chunk_seq == 41 and values == [1.0 + i for i in range(12)]
    assert (seen[0].delay, seen[0].executed, seen[0].reason) == (3, 7, 2)


def test_the_policy_sees_the_unplayed_remainder():
    seen = []
    server.answer(lambda r: seen.append(r.previous) or [0.0] * 6, request_bytes(previous=[0.5] * 12))
    assert seen == [[0.5] * 12]


@pytest.mark.parametrize("plan", [[0.0] * 7, [0.0] * 306])
def test_a_plan_that_is_not_whole_steps_or_too_long_is_refused(plan):
    with pytest.raises(wire.WireError):
        server.answer(lambda r: plan, request_bytes())


def test_a_request_that_does_not_decode_does_not_reach_the_policy():
    asked = []
    with pytest.raises(wire.WireError) as e:
        server.answer(lambda r: asked.append(r) or [0.0] * 6, request_bytes()[:-1])
    assert e.value.kind == "truncated" and not asked
