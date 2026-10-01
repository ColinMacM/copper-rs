# cu-policy

A policy loop for Copper: a policy process, such as a vision-language-action model, proposes
chunks of future actions, and a governor inside the Copper graph gates them before they reach an
arm.

The crate contains:

- `wire`: the byte format of every message between the graph and the policy process. It has no
  dependencies and builds without `std`.
- `governor::ActionGovernor`: a `CuTask` that checks, limits and plays action chunks, and the
  chunk scheduler that decides when the policy is asked again.
- `link`: the Zenoh bridge to the policy process (feature `link`).
- `python/copper_policy`: the Python package that serves a policy over Zenoh.

## Python package

```bash
pip install ./components/tasks/cu_policy            # zenoh, numpy
pip install "./components/tasks/cu_policy[flow]"    # adds torch: real-time chunking, flow policies
pip install "./components/tasks/cu_policy[act]"     # adds torch and lerobot: the ACT policy
python -m copper_policy --connect-port 7447 --key-prefix vla
```

The wheel is pure Python.
