"""Policies the runner can serve. A policy maps (seq, raw_state) to a list of steps, each a
list of JOINTS raw-tick goals."""
import math
import time

STEP_S = 1 / 30


class Scripted:
    """A sinusoid around the start position, for loop tests."""

    def __init__(self, amplitude=400.0, freq_hz=0.4, steps=20, target=None):
        self.amplitude, self.freq, self.steps = amplitude, freq_hz, steps
        self.fixed_target = target
        self.t0 = time.monotonic()
        self.center = None

    def __call__(self, seq, state):
        if self.center is None:
            self.center = list(state[:6])
        now = time.monotonic() - self.t0
        out = []
        for k in range(self.steps):
            t = now + (k + 1) * STEP_S
            wave = self.amplitude * math.sin(2 * math.pi * self.freq * t)
            out.append([self.fixed_target if self.fixed_target is not None else c + wave for c in self.center])
        return out


class Act:
    """A LeRobot ACT policy: raw ticks -> LeRobot units -> predict_action_chunk -> raw ticks.

    `policy` is a constructed `ACTPolicy` (random weights are fine for loop tests); `motors` come
    from `calibration.load`. One camera-less observation is fed: a state vector and, if the policy
    expects images, `image_fn()` must return a float CHW tensor in [0, 1].
    """

    def __init__(self, policy, motors, image_fn=None, device="cpu", env_state=False):
        import torch  # imported here so the runner works without torch installed

        self.torch, self.policy, self.motors = torch, policy, motors
        self.image_fn, self.device, self.env_state = image_fn, device, env_state
        policy.eval()

    def __call__(self, seq, state):
        from . import calibration
        from .wire import MAX_CHUNK_VALUES, JOINTS

        torch = self.torch
        units = calibration.state_to_units(self.motors, state[: len(self.motors)])
        batch = {"observation.state": torch.tensor([units], dtype=torch.float32, device=self.device)}
        if self.env_state:
            batch["observation.environment_state"] = batch["observation.state"].clone()
        if self.image_fn is not None:
            batch["observation.images.top"] = self.image_fn().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)[0].cpu().numpy()
        # A chunk on the wire holds at most MAX_CHUNK_VALUES // JOINTS steps; later steps would
        # be stale by the time they could run anyway.
        steps = chunk[: MAX_CHUNK_VALUES // JOINTS]
        return [
            [calibration.units_to_raw(m, float(v)) for m, v in zip(self.motors, step)]
            for step in steps
        ]


def build_act(calibration_path, modes, device="cpu", seed=0):
    """An ACT policy with random weights and a state-only input, for loop tests. A trained
    checkpoint is loaded with `ACTPolicy.from_pretrained` instead and passed to `Act` the same way."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    from . import calibration

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (6,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        pretrained_backbone_weights=None,
        device=device,
        chunk_size=100,
        n_action_steps=100,
    )
    policy = ACTPolicy(cfg).to(device)
    motors = calibration.load(calibration_path, modes)
    return Act(policy, motors, device=device, env_state=True)
