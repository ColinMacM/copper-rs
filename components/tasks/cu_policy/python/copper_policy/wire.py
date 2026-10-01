"""The wire format of the policy link, in Python.

The byte layout of every message is defined in `src/wire.rs` of the `cu-policy` crate, and
`tests/golden/vectors.json` holds vectors that both implementations are tested against. All
integers are little-endian and fixed-width.

A codec raises `WireError`, a `ValueError`, whose `kind` is `"truncated"` (the input ends early or
has bytes left over), `"too_long"` (a length above the capacity of the message) or
`"partial_step"` (a chunk that is not a whole number of steps).
"""
import collections
import struct

JOINTS = 6
MAX_STEPS = 50
MAX_CHUNK_VALUES = JOINTS * MAX_STEPS
OBS_JOINTS = 8

_U32 = struct.Struct("<I")
_CHUNK_HEADER = struct.Struct("<QI")
_OBS_HEADER = struct.Struct("<QQI")
_IMAGE_HEADER = struct.Struct("<QQIII4sI")
IMAGE_HEADER_BYTES = _IMAGE_HEADER.size


class WireError(ValueError):
    """A message that fails to decode or encode. `kind` is "truncated", "too_long" or "partial_step"."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def _truncated(what):
    return WireError("truncated", f"{what} ends early or has bytes left over")


def _floats(data, pos, count, capacity, what):
    """`count` floats at `pos`, after checking the capacity and that they are all there."""
    if count > capacity:
        raise WireError("too_long", f"{what} announces {count} values, the capacity is {capacity}")
    end = pos + 4 * count
    if end > len(data):
        raise _truncated(what)
    return list(struct.unpack_from(f"<{count}f", data, pos)), end


def _pack_floats(values, capacity, what):
    values = [float(v) for v in values]
    if len(values) > capacity:
        raise WireError("too_long", f"{len(values)} {what} values exceed the capacity {capacity}")
    return struct.pack(f"<{len(values)}f", *values)


def _finish(data, end, what):
    if end != len(data):
        raise _truncated(what)


def decode_obs(data):
    """An observation: `(seq, tov_ns, [state floats])`."""
    if len(data) < _OBS_HEADER.size:
        raise _truncated("an observation")
    seq, tov_ns, n = _OBS_HEADER.unpack_from(data)
    state, end = _floats(data, _OBS_HEADER.size, n, OBS_JOINTS, "an observation")
    _finish(data, end, "an observation")
    return seq, tov_ns, state


def encode_obs(seq, state, tov_ns=0):
    return _OBS_HEADER.pack(seq, tov_ns, len(state)) + _pack_floats(state, OBS_JOINTS, "state")


def decode_chunk(data):
    """An action chunk: `(obs_seq, [values])`."""
    if len(data) < _CHUNK_HEADER.size:
        raise _truncated("a chunk")
    seq, n = _CHUNK_HEADER.unpack_from(data)
    values, end = _floats(data, _CHUNK_HEADER.size, n, MAX_CHUNK_VALUES, "a chunk")
    _finish(data, end, "a chunk")
    return seq, values


def encode_chunk(obs_seq, values):
    """`values`: flat iterable of floats, a whole number of steps. Sent as given: the governor
    rejects non-finite and out-of-range values."""
    values = list(values)
    if len(values) % JOINTS:
        raise WireError("partial_step", f"{len(values)} values is not a whole number of {JOINTS}-joint steps")
    return _CHUNK_HEADER.pack(obs_seq, len(values)) + _pack_floats(values, MAX_CHUNK_VALUES, "chunk")


def decode_image_header(data):
    """The header of a camera frame: `(seq, tov_ns, width, height, stride, pixel_format, len)`,
    from the start of `data`; `len` bytes of pixels follow it."""
    if len(data) < _IMAGE_HEADER.size:
        raise _truncated("an image header")
    return _IMAGE_HEADER.unpack_from(data)


def encode_image_header(seq, tov_ns, width, height, stride, pixel_format, length):
    return _IMAGE_HEADER.pack(seq, tov_ns, width, height, stride, pixel_format, length)


def decode_image(data):
    """A camera frame: `(seq, tov_ns, width, height, stride, pixel_format, pixels)`."""
    seq, tov_ns, w, h, stride, fmt, n = decode_image_header(data)
    if len(data) != _IMAGE_HEADER.size + n or n != stride * h:
        raise WireError("truncated", f"image length {len(data)} does not match a {stride}x{h} frame of {n} bytes")
    return seq, tov_ns, w, h, stride, fmt, data[_IMAGE_HEADER.size:]


def frame_pattern(seq, n):
    """The synthetic camera's pixels: byte i is (i + seq*31) % 256."""
    off = (seq % 256) * 31 % 256
    reps = (off + n) // 256 + 1
    return (bytes(range(256)) * reps)[off:off + n]


def check_frame(data, stats):
    """Counts a frame as good only if every pixel is what the camera painted for its seq."""
    try:
        seq, tov_ns, _w, _h, _stride, _fmt, pixels = decode_image(data)
    except ValueError:
        stats["bad_frames"] += 1
        return
    if pixels == frame_pattern(seq, len(pixels)):
        stats["frames"] += 1
        stats["last_frame_tov"] = tov_ns
    else:
        stats["bad_frames"] += 1


_EXEC = struct.Struct("<QQIIIIf")
HAS_STAMP, CHUNK_ACTIVE, PLAYED, ACCEPTED = 1, 2, 4, 8
REASON_FIRST, REASON_SCHEDULED, REASON_EVENT = 1, 2, 4


class Exec(
    collections.namedtuple(
        "Exec",
        "stamp_seq chunk_seq next_index flags accept_skip reject tracking_err",
        defaults=(0, 0, 0.0),
    )
):
    """What the governor executes: the active chunk and the step that plays next, the step an
    accepted chunk started at (its real delay), why a chunk was refused, and how far the
    measurement is from the target played last cycle."""

    __slots__ = ()

    @property
    def has_stamp(self):
        return bool(self.flags & HAS_STAMP)

    @property
    def chunk_active(self):
        return bool(self.flags & CHUNK_ACTIVE)

    @property
    def played(self):
        return bool(self.flags & PLAYED)

    @property
    def accepted(self):
        return bool(self.flags & ACCEPTED)


def decode_exec(data):
    if len(data) != _EXEC.size:
        raise _truncated("an exec state")
    return Exec(*_EXEC.unpack(data))


def encode_exec(stamp_seq, chunk_seq, next_index, flags, accept_skip=0, reject=0, tracking_err=0.0):
    return _EXEC.pack(stamp_seq, chunk_seq, next_index, flags, accept_skip, reject, tracking_err)


MODE_NAIVE, MODE_RTC = 0, 1
FLAG_PROJECT, FLAG_ROLL_OBS, FLAG_POSITIONAL_NOISE = 1, 2, 4

_REQ = struct.Struct("<QIIIIIIIIfI")
Request = collections.namedtuple(
    "Request",
    "obs_seq delay executed reason horizon mode denoise_steps best_of flags beta state previous",
)


class Options(
    collections.namedtuple(
        "Options",
        "horizon mode denoise_steps best_of flags beta",
        defaults=(MAX_STEPS, MODE_NAIVE, 5, 1, 0, 5.0),
    )
):
    """How the policy is to plan, as the governor sends it with every request: the prediction
    horizon, `MODE_NAIVE` or `MODE_RTC`, the denoising steps of a flow policy, the guided samples
    drawn per chunk, `FLAG_*` bits and the clip of the guidance weight."""

    __slots__ = ()


def decode_request(data):
    """An `InferenceRequest` from the governor's scheduler: the observation, the delay estimate
    `d`, the steps `s` of the active chunk already played, how the policy is to plan, and the
    unplayed remainder of the active chunk (flat, 6 values per step, empty when nothing is
    executing)."""
    if len(data) < _REQ.size:
        raise _truncated("a request")
    obs_seq, delay, executed, reason, horizon, mode, steps, best_of, flags, beta, n = _REQ.unpack_from(data)
    state, pos = _floats(data, _REQ.size, n, OBS_JOINTS, "a request's state")
    if pos + _U32.size > len(data):
        raise _truncated("a request")
    (m,) = _U32.unpack_from(data, pos)
    previous, end = _floats(data, pos + _U32.size, m, MAX_CHUNK_VALUES, "a request's previous chunk")
    _finish(data, end, "a request")
    return Request(obs_seq, delay, executed, reason, horizon, mode, steps, best_of, flags, beta, state, previous)


def encode_request(obs_seq, delay, executed, reason, state, previous, options=Options()):
    return (
        _REQ.pack(obs_seq, delay, executed, reason, *options, len(state))
        + _pack_floats(state, OBS_JOINTS, "state")
        + _U32.pack(len(previous))
        + _pack_floats(previous, MAX_CHUNK_VALUES, "previous")
    )
