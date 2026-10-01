import pytest
import torch

from vla_runner import flow_policy as fp

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def policy():
    """The demo flow policy, trained once per test session (about 15 s)."""
    return fp.FlowPolicy(fp.train(3000))
