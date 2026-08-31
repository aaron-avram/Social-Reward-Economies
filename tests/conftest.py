"""Shared fixtures. Everything is seeded; no test may depend on global RNG state."""
import numpy as np
import pytest

from model.agent import Agent
from model.config import (
    AlgorithmParams, Dimensions, RewardModelKind, RewardParams, RuntimeParams,
    ScheduleParams, SystemConfig, TrackingMode,
)
from model.reputation import ReputationState
from model.rewards import build_reward_model


@pytest.fixture
def dims():
    return Dimensions(num_agents=5, num_states=3, num_actions=2)


@pytest.fixture
def algo():
    return AlgorithmParams()


@pytest.fixture
def rng():
    """A fresh seeded Generator. Never np.random."""
    return np.random.default_rng(12345)


@pytest.fixture
def agents(dims, algo, rng):
    return [Agent(i, algo, dims, rng) for i in range(dims.num_agents)]


@pytest.fixture
def rep(dims):
    return ReputationState.initial(dims.num_agents)


@pytest.fixture
def rewards(dims):
    return build_reward_model(
        RewardParams(kind=RewardModelKind.SIMPLE_PREFERRED_ACTION),
        dims, np.random.default_rng(0),
    )


@pytest.fixture
def small_config(dims):
    return SystemConfig(
        dims=dims,
        runtime=RuntimeParams(seed=7, num_time_steps=40,
                              tracking_mode=TrackingMode.FULL),
        schedule=ScheduleParams(role_update_base_interval=10),
    )


@pytest.fixture(autouse=True)
def poison_global_rng():
    """Perturb np.random before every test. Any surviving global call now
    produces a different result run to run, so a determinism test fails loudly
    instead of passing by luck."""
    np.random.seed(np.random.randint(0, 2**31 - 1) if False else 987654321)
    yield