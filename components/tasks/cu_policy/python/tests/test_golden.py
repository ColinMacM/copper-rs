"""The wire vectors in `tests/golden/vectors.json`, the same file the Rust tests read. They were
written by an independent encoder: every vector must encode to its bytes and decode from them,
and every rejected input must fail with its error kind."""
import json
import pathlib

import pytest

from copper_policy import wire

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = json.loads((ROOT / "tests" / "golden" / "vectors.json").read_text())


def encode(kind, f):
    if kind == "obs":
        return wire.encode_obs(f["seq"], f["state"], f["tov_ns"])
    if kind == "chunk":
        return wire.encode_chunk(f["obs_seq"], f["values"])
    if kind == "exec":
        return wire.encode_exec(**f)
    if kind == "request":
        options = wire.Options(f["horizon"], f["mode"], f["denoise_steps"], f["best_of"], f["flags"], f["beta"])
        return wire.encode_request(
            f["obs_seq"], f["delay"], f["executed"], f["reason"], f["state"], f["previous"], options
        )
    if kind == "image_header":
        return wire.encode_image_header(
            f["seq"], f["tov_ns"], f["width"], f["height"], f["stride"], f["pixel_format"].encode(), f["len"]
        )
    raise AssertionError(kind)


def decode(kind, data):
    """The decoded message as a dict shaped like the vector's fields."""
    if kind == "obs":
        seq, tov_ns, state = wire.decode_obs(data)
        return {"seq": seq, "tov_ns": tov_ns, "state": state}
    if kind == "chunk":
        seq, values = wire.decode_chunk(data)
        return {"obs_seq": seq, "values": values}
    if kind == "exec":
        return dict(wire.decode_exec(data)._asdict())
    if kind == "request":
        r = wire.decode_request(data)
        return {"obs_seq": r.obs_seq, "delay": r.delay, "executed": r.executed, "reason": r.reason,
                "horizon": r.horizon, "mode": r.mode, "denoise_steps": r.denoise_steps, "best_of": r.best_of,
                "flags": r.flags, "beta": r.beta, "state": r.state, "previous": r.previous}
    if kind == "image_header":
        seq, tov_ns, w, h, stride, fmt, n = wire.decode_image_header(data)
        return {"seq": seq, "tov_ns": tov_ns, "width": w, "height": h, "stride": stride,
                "pixel_format": fmt.decode(), "len": n}
    raise AssertionError(kind)


@pytest.mark.parametrize("vector", DOC["vectors"], ids=lambda v: v["name"])
def test_vector_encodes_to_its_bytes_and_decodes_from_them(vector):
    want = bytes.fromhex(vector["hex"])
    assert encode(vector["kind"], vector["fields"]) == want
    assert decode(vector["kind"], want) == vector["fields"]


@pytest.mark.parametrize("case", DOC["reject"], ids=lambda c: c["name"])
def test_rejected_input_fails_with_its_error_kind(case):
    with pytest.raises(wire.WireError) as e:
        decode(case["kind"], bytes.fromhex(case["hex"]))
    assert e.value.kind == case["error"]


def test_every_message_kind_has_vectors_and_rejections():
    kinds = {v["kind"] for v in DOC["vectors"]}
    assert kinds == {"obs", "chunk", "exec", "request", "image_header"}
    assert kinds <= {c["kind"] for c in DOC["reject"]} | {"exec"}


def test_bytes_left_over_after_a_message_are_an_error():
    good = wire.encode_chunk(3, [0.0] * 6)
    with pytest.raises(wire.WireError) as e:
        wire.decode_chunk(good + b"\x00")
    assert e.value.kind == "truncated"
