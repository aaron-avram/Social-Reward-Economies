"""Shared fixtures. Everything is seeded; no test may depend on global RNG state."""
import numpy as np
import pytest

from model.agent import Agent
from model.config import (
    AlgorithmParams, Dimensions, ScheduleParams,
    RuntimeParams, SystemConfig, TrackingMode,
)
from model.reputation import ReputationState


def config_field(cfg):
    """SystemConfig's dimensions field is named `dimensions`, but every consumer
    reads `.dims`. Tests go through this so the suite reports ONE failure for the
    naming mismatch rather than failing everywhere."""
    return getattr(cfg, "dims", None) or cfg.dimensions


@pytest.fixture
def dims():
    return Dimensions(num_agents=5, num_states=3, num_actions=2)


@pytest.fixture
def algo():
    return AlgorithmParams()


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def agents(dims, algo, rng):
    return [Agent(i, algo, dims, rng) for i in range(dims.num_agents)]


@pytest.fixture
def rep(dims):
    return ReputationState.initial(dims.num_agents)


@pytest.fixture
def small_config():
    return SystemConfig(
        dims=Dimensions(num_agents=5, num_states=3, num_actions=2),
        runtime=RuntimeParams(seed=7, num_time_steps=40, tracking_mode=TrackingMode.FULL),
        schedule=ScheduleParams(role_update_base_interval=10),
    )
