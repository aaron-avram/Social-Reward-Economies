"""config.py — dataclass wiring, validation, enum round-trip."""
import json
from dataclasses import FrozenInstanceError, replace, fields

import pytest

from model import config as C


def test_instances_are_independent():
    """The classic dataclass trap: a mutable default shared across instances would
    make a parameter sweep contaminate every config alive in the process."""
    a, b = C.SystemConfig(), C.SystemConfig()
    assert a.algorithm is not b.algorithm
    assert a.stepsizes is not b.stepsizes
    assert a.stepsizes.alpha_pu is not b.stepsizes.alpha_pu


def test_nested_groups_are_dataclass_fields():
    names = {f.name for f in fields(C.SystemConfig)}
    assert {"algorithm", "reward", "stepsizes", "runtime", "schedule"} <= names


def test_dimensions_attribute_is_named_dims():
    """Every consumer (system.py, plots.py) reads config.dims."""
    assert hasattr(C.SystemConfig(), "dims")


def test_reward_params_expose_kind():
    """rewards.build_reward_model indexes the registry with params.kind."""
    assert hasattr(C.RewardParams(), "kind")


@pytest.mark.parametrize("kwargs", [
    dict(B_F=0.9, B_R=0.8),      # hysteresis inverted
    dict(B_F=-0.1),              # out of range
    dict(c_threshold=1.5),
    dict(gamma=-1.0),
])
def test_algorithm_params_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        C.AlgorithmParams(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(order_gap=-0.1),
    dict(order_gap=10.0, clip_min=0.01, clip_max=2.5),
])
def test_reward_params_reject_invalid(kwargs):
    """__post_init__, not __post__init__ — a typo makes validation silently dead."""
    with pytest.raises(ValueError):
        C.RewardParams(**kwargs)


def test_stepsize_matches_original_schedule():
    """Values and decays from step() 1899-1907 of code_debugged.py."""
    s = C.StepsizeParams().at(2000)
    assert s.alpha_pu == pytest.approx(0.05 / (1 + 2000 * 0.01))
    assert s.beta_status == pytest.approx(0.10 / (1 + 2000 * 0.01))
    assert s.eta_v == pytest.approx(0.10 / (1 + 2000 * 0.01))
    assert s.eta_s == pytest.approx(0.10 / (1 + 2000 * 0.01))
    assert s.eta_J == pytest.approx(0.05 / (1 + 2000 * 0.01))
    assert s.alpha_rate == pytest.approx(0.01 / (1 + 2000 * 0.005))


def test_stepsize_clamps_t_at_one():
    assert C.StepsizeParams().at(0) == C.StepsizeParams().at(1)


def test_config_dict_is_json_serialisable():
    """to_dict feeds the results file alongside SCHEMA_VERSION; a raw Enum breaks it."""
    d = C.SystemConfig().to_dict()
    json.dumps(d)
    assert d["schema_version"] == C.SCHEMA_VERSION


def test_config_dict_unwraps_enums():
    d = C.SystemConfig().to_dict()
    assert d["reward"]["model"] == "simple_preferred_action"
    assert d["runtime"]["tracking_mode"] == "full"


def test_param_groups_are_frozen_or_replace_is_deep():
    """A sweep that mutates cfg.algorithm.gamma must not affect the base config.
    Either the groups are frozen (mutation raises) or replace() deep-copies."""
    base = C.SystemConfig()
    derived = replace(base, dimensions=base.dims) if hasattr(base, "dims") else replace(base)
    try:
        derived.algorithm.gamma = 99.0
    except FrozenInstanceError:
        return
    assert base.algorithm.gamma != 99.0, "mutating a derived config leaked into the base"
