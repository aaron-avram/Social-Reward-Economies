"""agent.py — Agent must be a leaf: no system reference, no RNG after construction."""
import numpy as np
import pytest

from model.agent import Agent, AgentRole
from model.config import ActorRateDriverMode, AlgorithmParams, Dimensions


def _agent(agent_id=0, **kw):
    return Agent(agent_id, AlgorithmParams(**kw),
                 Dimensions(num_agents=5, num_states=3, num_actions=2),
                 np.random.default_rng(0))


def test_weights_match_configured_dimensions():
    """AgentState's defaults must not hardcode (3, 2)."""
    a = Agent(0, AlgorithmParams(), Dimensions(num_agents=4, num_states=7, num_actions=5),
              np.random.default_rng(0))
    assert a.state.weights_pu.shape == (7, 5)
    assert a.state.weights_status.shape == (7, 5)


def test_construction_is_seeded():
    d = Dimensions()
    a = Agent(0, AlgorithmParams(), d, np.random.default_rng(4))
    b = Agent(0, AlgorithmParams(), d, np.random.default_rng(4))
    assert np.array_equal(a.state.weights_pu, b.state.weights_pu)


def test_agent_holds_no_system_reference():
    assert not hasattr(_agent(), "system")


def test_softmax_policy_is_a_distribution():
    a = _agent()
    p = a.get_softmax_policy(0, a.state.weights_pu)
    assert p.shape == (2,)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
    assert (p >= 0).all()


def test_select_action_is_inverse_cdf_of_the_uniform():
    """u below the first action's probability must select action 0, and u above it
    action 1. This is what makes the draw a function of (seed, agent, t)."""
    a = _agent()
    a.state.weights_pu = np.array([[10.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    p0 = a.get_softmax_policy(0, a.state.weights_pu)[0]
    assert a.select_action(0, p0 * 0.5, None) == 0
    assert a.select_action(0, min(p0 + (1 - p0) * 0.5, 0.999999), None) == 1


def test_select_action_never_returns_out_of_range():
    """The softmax divides by (sum + 1e-8), so the CDF stops just short of 1.0;
    without renormalisation u near 1 indexes past the last action."""
    a = _agent()
    for u in (0.0, 0.5, 1 - 1e-12, 0.9999999999999):
        assert 0 <= a.select_action(0, u, None) < 2


def test_select_action_returns_python_int():
    """np.int64 is not JSON-serialisable and these land in audit rows."""
    assert type(_agent().select_action(0, 0.5, None)) is int


def test_current_policy_uses_leader_weights_when_following():
    a = _agent()
    a.state.role = AgentRole.REPUTATION
    a.state.following = 1
    leader = np.array([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(a.get_current_policy(0, leader), a.get_softmax_policy(0, leader))


def test_current_policy_uses_status_weights_when_status():
    a = _agent()
    a.state.role = AgentRole.STATUS
    a.state.weights_status = np.ones((3, 2))
    assert np.allclose(a.get_current_policy(0, None),
                       a.get_softmax_policy(0, a.state.weights_status))


def test_adopt_behavior_ignores_unset_leader():
    """Guard from line 419 of the original: no leader means no copy."""
    a = _agent()
    before = a.state.weights_pu.copy()
    a.state.following = None
    a.adopt_behavior(None)
    assert np.array_equal(a.state.weights_pu, before)


def test_adopt_behavior_ignores_self_follow():
    a = _agent(agent_id=0)
    before = a.state.weights_pu.copy()
    a.state.following = 0
    a.adopt_behavior(np.ones((3, 2)))
    assert np.array_equal(a.state.weights_pu, before)


def test_adopt_behavior_copies_not_aliases():
    a = _agent()
    a.state.following = 2
    leader = np.ones((3, 2))
    a.adopt_behavior(leader)
    leader[0, 0] = 99.0
    assert a.state.weights_pu[0, 0] == 1.0


def test_update_personal_utility_does_not_append_payoff_history():
    """The duplicate append at line 264 double-weighted PU periods in the
    trajectory mean. system._actors_act owns the single append."""
    a = _agent()
    a.update_personal_utility(0, 0, 1.0, 0.05, 0.05)
    assert a.state.payoff_history == []


def test_policy_gradient_moves_toward_the_rewarded_action():
    a = _agent()
    w = np.zeros((3, 2))
    new = a.update_policy_gradient(0, 1, reward=1.0, weights=w, lr=0.5)
    assert new[0, 1] > new[0, 0]
    assert np.array_equal(w, np.zeros((3, 2))), "must not mutate the input weights"


def test_actor_rate_driver_is_the_weighted_max():
    a = _agent(gamma=2.0, kappa=3.0)
    a.state.estimated_reward_pu = 1.0
    a.state.estimated_reward_rep = 0.4     # 0.8 weighted
    a.state.estimated_reward_status = 0.5  # 1.5 weighted
    assert a.actor_rate_driver() == pytest.approx(1.5)


def test_status_override_uses_unweighted_status_estimate():
    """The override fires only when kappa == 0, so the weighted term is 0 and
    using it would defeat the mode."""
    a = _agent(kappa=0.0, actor_rate_driver_mode=ActorRateDriverMode.STATUS_IF_FOLLOWERS_KAPPA0,
               actor_rate_status_override_min_followers=2)
    a.state.estimated_reward_status = 0.7
    a.state.followers = {1, 2, 3}
    assert a.actor_rate_driver() == pytest.approx(0.7)


def test_status_override_requires_enough_followers():
    a = _agent(kappa=0.0, actor_rate_driver_mode=ActorRateDriverMode.STATUS_IF_FOLLOWERS_KAPPA0,
               actor_rate_status_override_min_followers=5)
    a.state.estimated_reward_pu = 0.2
    a.state.estimated_reward_status = 0.7
    a.state.followers = {1}
    assert a.actor_rate_driver() == pytest.approx(0.2)


def test_driver_matches_terms_breakdown():
    """The production scalar path and the audit dict path must not drift."""
    for kappa, followers in ((2.0, set()), (0.0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})):
        a = _agent(kappa=kappa,
                   actor_rate_driver_mode=ActorRateDriverMode.STATUS_IF_FOLLOWERS_KAPPA0)
        a.state.estimated_reward_pu = 0.3
        a.state.estimated_reward_rep = 0.2
        a.state.estimated_reward_status = 0.6
        a.state.followers = followers
        assert a.actor_rate_driver() == pytest.approx(a.actor_rate_terms()["driver"])


def test_rate_terms_mode_is_a_plain_string():
    """str(Enum) gives 'ActorRateDriverMode.STANDARD'; the export needs 'standard'."""
    assert _agent().actor_rate_terms()["actor_rate_driver_mode"] == "standard"


def test_actor_rate_stays_within_budget():
    a = _agent(M=1.0)
    a.state.estimated_reward_pu = 1e6
    for _ in range(50):
        a.update_actor_interaction_rate(0.5)
    assert 0.0 <= a.state.actor_interaction_rate <= 1.0
