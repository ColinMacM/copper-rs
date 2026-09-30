import math
import struct

import pytest

from vla_runner import wire


def test_obs_roundtrip_and_exact_layout():
    data = wire.encode_obs(2**40 + 7, [1.5, -2.0, 3.25], tov_ns=2**45 + 3)
    assert data == struct.pack("<QQI3f", 2**40 + 7, 2**45 + 3, 3, 1.5, -2.0, 3.25)
    seq, tov_ns, state = wire.decode_obs(data)
    assert (seq, tov_ns, state) == (2**40 + 7, 2**45 + 3, [1.5, -2.0, 3.25]), "64-bit fields stay exact"


@pytest.mark.parametrize("bad", [b"", b"\x00" * 19, struct.pack("<QQI", 1, 2, 3) + b"\x00" * 8])
def test_malformed_observations_are_rejected(bad):
    with pytest.raises(ValueError):
        wire.decode_obs(bad)


def test_chunk_layout_matches_the_rust_encoding():
    data = wire.encode_chunk(9, [0.0] * 12)
    assert data[:12] == struct.pack("<QI", 9, 12) and len(data) == 12 + 48
    assert wire.decode_chunk(data) == (9, [0.0] * 12)


def test_chunk_shape_is_enforced_but_values_are_not_filtered():
    with pytest.raises(ValueError):
        wire.encode_chunk(1, [0.0] * 7)
    with pytest.raises(ValueError):
        wire.encode_chunk(1, [0.0] * 306)
    # The runner does not hide bad values from the governor.
    _, values = wire.decode_chunk(wire.encode_chunk(1, [float("nan")] * 6))
    assert all(math.isnan(v) for v in values)


def _image(seq, w=4, h=2, pixels=None):
    stride = w * 3
    pixels = wire.frame_pattern(seq, stride * h) if pixels is None else pixels
    return struct.pack("<QQIII4sI", seq, 99, w, h, stride, b"RGB3", len(pixels)) + pixels


def test_image_layout_and_pattern():
    seq, tov, w, h, stride, fmt, pixels = wire.decode_image(_image(1000))
    assert (seq, tov, w, h, stride, fmt) == (1000, 99, 4, 2, 12, b"RGB3")
    off = 1000 % 256 * 31 % 256
    assert pixels == bytes((i + off) % 256 for i in range(24))


def test_truncated_or_inconsistent_images_are_rejected():
    good = _image(5)
    for bad in (b"", good[:-1], good + b"\x00", _image(5, pixels=b"\x00" * 23)):
        with pytest.raises(ValueError):
            wire.decode_image(bad)


def test_a_corrupted_frame_is_counted_bad():
    stats = {"frames": 0, "bad_frames": 0, "last_frame_tov": 0}
    wire.check_frame(_image(7), stats)
    flipped = bytearray(_image(7))
    flipped[-1] ^= 1
    wire.check_frame(bytes(flipped), stats)
    wire.check_frame(_image(7)[:-3], stats)
    assert (stats["frames"], stats["bad_frames"], stats["last_frame_tov"]) == (1, 2, 99)
