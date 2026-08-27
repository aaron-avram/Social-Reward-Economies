"""rewards.py — table construction, the ABC contract, and the registry."""
import numpy as np
import pytest

from model.config import Dimensions, RewardModelKind, RewardParams
from model import rewards as R


def _dims(a=5, s=3, act=2):
    return Dimensions(num_agents=a, num_states=s, num_actions=act)


@pytest.mark.parametrize("kind", list(RewardModelKind))
def test_every_kind_builds_a_well_shaped_table(kind):
    dims = _dims()
    model = R.REWARD_MODELS[kind](RewardParams(kind=kind), dims, np.random.default_rng(0))
    assert model.table.shape == (dims.num_agents, dims.num_states, dims.num_actions)
    assert np.isfinite(model.table).all()


def test_registry_covers_every_kind():
    """The enum and the registry live in different files and will drift."""
    assert set(R.REWARD_MODELS) == set(RewardModelKind)


def test_build_reward_model_dispatches():
    p = RewardParams(kind=RewardModelKind.SHARED_BASE_GAUSSIAN)
    m = R.build_reward_model(p, _dims(), np.random.default_rng(0))
    assert isinstance(m, R.SharedBaseGaussian)

def test_subclass_returning_wrong_shape_fails_at_construction():
    class Bad(R.RewardModel):
        def _build_table(self, params, dims, rng):
            return np.zeros((2, 2))
    with pytest.raises(TypeError):
        Bad(RewardParams(), _dims(), np.random.default_rng(0))


def test_simple_preferred_action_marks_agent_id_mod_actions():
    dims = _dims()
    m = R.SimplePreferredAction(RewardParams(), dims, np.random.default_rng(0))
    for i in range(dims.num_agents):
        pref = i % dims.num_actions
        for s in range(dims.num_states):
            assert m.table[i, s, pref] == 1.0
            assert m.table[i, s, 1 - pref] == 0.0


def test_shared_base_gaussian_shares_base_across_agents():
    """'Shared base' means m(s,a) is drawn ONCE over (states, actions). If it were
    drawn per agent the cross-agent correlation would vanish."""
    dims = _dims(a=200, s=3, act=2)
    p = RewardParams(base_sigma=0.5, agent_sigma=1e-6)
    m = R.SharedBaseGaussian(p, dims, np.random.default_rng(0))
    spread_within = m.table.std(axis=0).max()      # across agents, same (s,a)
    spread_across = m.table.mean(axis=0).std()     # across (s,a)
    assert spread_within < spread_across


def test_shared_good_bad_enforces_order_gap():
    dims = _dims(a=10, s=4, act=3)
    p = RewardParams(order_gap=0.05)
    m = R.SharedGoodBadHeterogeneous(p, dims, np.random.default_rng(1))
    good = m._shared_good_actions
    for i in range(dims.num_agents):
        for s in range(dims.num_states):
            g = int(good[s])
            bad = [a for a in range(dims.num_actions) if a != g]
            assert m.table[i, s, g] >= max(m.table[i, s, b] for b in bad) + p.order_gap - 1e-9


def test_consensus_welfare_requires_two_actions():
    with pytest.raises(ValueError):
        R.ConsensusWelfareGaussian(RewardParams(), _dims(act=3), np.random.default_rng(0))


def test_observer_utilities_matches_scalar_lookup():
    dims = _dims()
    m = R.SharedBaseGaussian(RewardParams(), dims, np.random.default_rng(0))
    vec = m.observer_utilities(1, 0)
    for i in range(dims.num_agents):
        assert vec[i] == m.observer_utility(i, 1, 0)


def test_observer_utilities_returns_a_copy():
    m = R.SharedBaseGaussian(RewardParams(), _dims(), np.random.default_rng(0))
    v = m.observer_utilities(0, 0)
    v[0] = 999.0
    assert m.table[0, 0, 0] != 999.0


def test_state_probabilities_uniform():
    p = R.state_probabilities(4)
    assert p.shape == (4,)
    assert p.sum() == pytest.approx(1.0)


def test_construction_is_deterministic_given_a_seed():
    d = _dims()
    p = RewardParams(kind=RewardModelKind.SHARED_GOOD_BAD_HETEROGENEOUS)
    a = R.SharedGoodBadHeterogeneous(p, d, np.random.default_rng(3)).table
    b = R.SharedGoodBadHeterogeneous(p, d, np.random.default_rng(3)).table
    assert np.array_equal(a, b)
