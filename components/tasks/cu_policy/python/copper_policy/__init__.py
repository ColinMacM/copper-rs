"""Serve a policy to a Copper policy loop.

`wire` is the byte format shared with the `cu-policy` crate, `runner` the Zenoh transport and the
command line (`python -m copper_policy`), `rtc` real-time chunking, `flow_policy` a small
flow-matching policy, `policies` the scripted and LeRobot ACT policies, `calibration` the
LeRobot calibration conversion. Importing the package needs only `zenoh` and `numpy`;
`rtc` and `flow_policy` need `torch` (extra `flow`) and the ACT policy needs `lerobot`
(extra `act`).
"""
