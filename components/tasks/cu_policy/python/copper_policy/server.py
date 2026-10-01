"""Serve a policy written in Python to the policy loop.

The governor's scheduler sends an inference request whenever the policy should be asked again:
the observation, the delay estimate, the steps of the active chunk already played and the
unplayed remainder of that chunk (`wire.Request`). A policy is a callable that turns a request
into a flat list of values, steps of `wire.JOINTS` values, the first of which belongs to the
control cycle of `request.obs_seq`; `serve` carries the requests and the answers over Zenoh.

    import math
    from copper_policy import server

    def policy(request):
        return [2048 + 400 * math.sin(0.08 * (request.obs_seq + i)) for i in range(20) for _ in range(6)]

    server.serve(policy, connect_port=7447, seconds=60, prefix="vla")

`prefix` is the instance name of the plugin, or the prefix of the routes configured on the
link's channels. The Rust counterpart is `cu_policy::server`.
"""
import threading
import time

from . import wire
from .runner import DEFAULT_KEY_PREFIX, Keys, session


def answer(policy, request_bytes):
    """Answers one request: decodes it, asks the policy and encodes the chunk, which names the
    observation of the request. Raises `wire.WireError` (a `ValueError`) for a request that does
    not decode or a plan that is not whole steps within the chunk capacity."""
    request = wire.decode_request(request_bytes)
    return wire.encode_chunk(request.obs_seq, policy(request))


def serve(policy, connect_port, seconds, *, prefix=DEFAULT_KEY_PREFIX, stop=None):
    """Answers requests for `seconds`, or until the `threading.Event` `stop` is set. Of the
    requests that arrive while the policy is busy only the newest is answered, so a slow policy
    works on current observations. Returns `{"requests", "chunks", "refused"}`."""
    keys = Keys(prefix)
    sess = session(connect_port=connect_port)
    pub = sess.declare_publisher(keys.action)
    lock = threading.Lock()
    wake = threading.Event()
    latest = {"request": None}

    def on_request(sample):
        with lock:
            latest["request"] = bytes(sample.payload)
        wake.set()

    sess.declare_subscriber(keys.infer, on_request)
    stats = {"requests": 0, "chunks": 0, "refused": 0}
    end = time.monotonic() + seconds
    while time.monotonic() < end and not (stop and stop.is_set()):
        wake.wait(0.05)
        wake.clear()
        with lock:
            data, latest["request"] = latest["request"], None
        if data is None:
            continue
        stats["requests"] += 1
        try:
            pub.put(answer(policy, data))
            stats["chunks"] += 1
        except ValueError:
            stats["refused"] += 1
    sess.close()
    return stats
