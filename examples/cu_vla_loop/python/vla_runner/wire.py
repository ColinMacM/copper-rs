"""Wire format shared with cu-policy-link (bincode, fixed-width little-endian integers).

ObsPacket   : seq u64 | tov_ns u64 | len u32 | len x f32
Image       : seq u64 | tov_ns u64 | width u32 | height u32 | stride u32 | pixel_format [4]u8 | len u32 | len x u8
ExecState   : stamp_seq u64 | chunk_seq u64 | next_index u32 | flags u32
ActionChunk : obs_seq u64 | len u32 | len x f32   (row-major, JOINTS values per step)
"""
import collections
import struct

JOINTS = 6
MAX_CHUNK_VALUES = 300
_HEADER = struct.Struct("<QI")
_OBS_HEADER = struct.Struct("<QQI")
_IMAGE_HEADER = struct.Struct("<QQIII4sI")


def decode_image(data):
    """Returns (seq, tov_ns, width, height, stride, pixel_format, pixels) or raises ValueError."""
    if len(data) < _IMAGE_HEADER.size:
        raise ValueError("image shorter than its header")
    seq, tov_ns, w, h, stride, fmt, n = _IMAGE_HEADER.unpack_from(data)
    if len(data) != _IMAGE_HEADER.size + n or n != stride * h:
        raise ValueError(f"image length {len(data)} does not match a {stride}x{h} frame of {n} bytes")
    return seq, tov_ns, w, h, stride, fmt, data[_IMAGE_HEADER.size:]


def frame_pattern(seq, n):
    """The synthetic camera's pixels: byte i is (i + seq*31) % 256."""
    off = (seq % 256) * 31 % 256
    reps = (off + n) // 256 + 1
    return (bytes(range(256)) * reps)[off:off + n]


def decode_obs(data):
    """Returns (seq, tov_ns, [state floats]) or raises ValueError on a malformed packet."""
    if len(data) < _OBS_HEADER.size:
        raise ValueError("observation shorter than its header")
    seq, tov_ns, n = _OBS_HEADER.unpack_from(data)
    if len(data) != _OBS_HEADER.size + 4 * n:
        raise ValueError(f"observation length {len(data)} does not match {n} values")
    return seq, tov_ns, list(struct.unpack_from(f"<{n}f", data, _OBS_HEADER.size))


def encode_chunk(obs_seq, values):
    """`values`: flat iterable of floats, a whole number of steps. Sent as given: rejecting
    non-finite or out-of-range values is the governor's job, not the runner's."""
    values = [float(v) for v in values]
    if len(values) % JOINTS:
        raise ValueError(f"{len(values)} values is not a whole number of {JOINTS}-joint steps")
    if len(values) > MAX_CHUNK_VALUES:
        raise ValueError(f"{len(values)} values exceeds the chunk capacity {MAX_CHUNK_VALUES}")
    return _HEADER.pack(obs_seq, len(values)) + struct.pack(f"<{len(values)}f", *values)


def decode_chunk(data):
    seq, n = _HEADER.unpack_from(data)
    return seq, list(struct.unpack_from(f"<{n}f", data, _HEADER.size))


def encode_obs(seq, state, tov_ns=0):
    state = [float(v) for v in state]
    return _OBS_HEADER.pack(seq, tov_ns, len(state)) + struct.pack(f"<{len(state)}f", *state)


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


_EXEC = struct.Struct("<QQII")
HAS_STAMP, CHUNK_ACTIVE, PLAYED = 1, 2, 4


class Exec(collections.namedtuple("Exec", "stamp_seq chunk_seq next_index flags")):
    """What the governor executes: the active chunk and the step that plays next."""

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


def decode_exec(data):
    if len(data) != _EXEC.size:
        raise ValueError(f"exec state is {len(data)} bytes, expected {_EXEC.size}")
    return Exec(*_EXEC.unpack(data))


def encode_exec(stamp_seq, chunk_seq, next_index, flags):
    return _EXEC.pack(stamp_seq, chunk_seq, next_index, flags)
