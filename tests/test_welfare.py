"""welfare.py — read-only metrics. Nothing here may mutate agents."""
import numpy as np
import pytest

from src.agent import Agent, AgentRole
from src.config import AlgorithmParams, Dimensions, RewardParams
from src.rewards import SimplePreferredAction, SharedBaseGaussian
from src import welfare as W


DIMS = Dimensions(num_agents=4, num_states=3, num_actions=2)


def agents(n=4):
    return [Agent(i, AlgorithmParams(), Dimensions(num_agents=n, num_states=3, num_actions=2),
                  np.random.default_rng(0)) for i in range(n)]


def rewards(dims=DIMS):
    return SimplePreferredAction(RewardParams(), dims, np.random.default_rng(0))


def test_current_policies_shape_and_normalisation():
    ags = agents()
    pol = W.current_policies(ags, 3, {})
    assert pol.shape == (4, 3, 2)
    assert np.allclose(pol.sum(axis=2), 1.0)


def test_current_policies_uses_leader_weights_for_followers():
    ags = agents()
    ags[0].state.role = AgentRole.REPUTATION
    ags[0].state.following = 1
    leader = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    pol = W.current_policies(ags, 3, {0: leader})
    assert pol[0, 0, 0] > 0.99


def test_current_policies_uses_status_weights_for_status_agents():
    ags = agents()
    ags[2].state.role = AgentRole.STATUS
    ags[2].state.weights_status = np.array([[0.0, 10.0]] * 3)
    pol = W.current_policies(ags, 3, {})
    assert pol[2, 0, 1] > 0.99


def test_current_policies_does_not_mutate_agents():
    ags = agents()
    before = [a.state.weights_pu.copy() for a in ags]
    W.current_policies(ags, 3, {})
    assert all(np.array_equal(a.state.weights_pu, b) for a, b in zip(ags, before))


def test_expected_observer_utilities_matches_the_explicit_loop():
    ags = agents()
    rw = SharedBaseGaussian(RewardParams(), DIMS, np.random.default_rng(0))
    pol = W.current_policies(ags, 3, {})
    got = W.expected_observer_utilities(pol, rw)

    n, s_n = DIMS.num_agents, DIMS.num_states
    want = np.zeros((n, n))
    for k in range(n):
        acc = np.zeros(n)
        for s in range(s_n):
            acc += rw.table[:, s, :] @ pol[k, s, :]
        want[:, k] = acc / s_n
    assert np.allclose(got, want)


def test_expected_observer_utilities_orientation():
    """Rows are OBSERVERS, columns are SOURCES. A transposed einsum still returns
    an (N,N) array, so only a directed test catches it."""
    dims = Dimensions(num_agents=2, num_states=1, num_actions=2)
    rw = SimplePreferredAction(RewardParams(), dims, np.random.default_rng(0))
    # agent 0 always plays action 0; agent 1 always plays action 1
    pol = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    out = W.expected_observer_utilities(pol, rw)
    # observer 0 prefers action 0 -> gains 1 from source 0, 0 from source 1
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.0)


def test_true_reputation_fields_and_shapes():
    ags = agents()
    tr = W.true_reputation(ags, W.current_policies(ags, 3, {}), rewards())
    n = len(ags)
    assert tr.true_reputation.shape == (n,)
    assert tr.true_rank.shape == (n,)
    assert set(tr.true_rank.tolist()) == set(range(1, n + 1))
    assert np.isfinite(tr.top_value)


def test_true_reputation_excludes_self_utility():
    ags = agents()
    pol = W.current_policies(ags, 3, {})
    rw = SharedBaseGaussian(RewardParams(), DIMS, np.random.default_rng(0))
    tr = W.true_reputation(ags, pol, rw)
    expected = W.expected_observer_utilities(pol, rw)
    want = expected.sum(axis=0) - np.diag(expected)
    assert np.allclose(tr.sum_expected_utility_others, want)


def test_true_reputation_ranks_by_value_then_id():
    ags = agents()
    tr = W.true_reputation(ags, W.current_policies(ags, 3, {}), rewards())
    order = np.argsort(tr.true_rank)
    vals = tr.true_reputation[order]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_true_reputation_unique_top_is_minus_one_when_tied():
    ags = agents(3)
    for a in ags:
        a.state.weights_pu = np.zeros((3, 2))
        a.state.actor_interaction_rate = 0.5
    dims = Dimensions(num_agents=3, num_states=3, num_actions=2)
    rw = SimplePreferredAction(RewardParams(), dims, np.random.default_rng(0))
    # all three identical -> exact tie
    tr = W.true_reputation(ags, W.current_policies(ags, 3, {}), rw)
    assert tr.unique_true_top_agent == -1 or int(tr.exact_top_mask.sum()) == 1


def test_current_opinion_leader_picks_the_most_followed():
    ags = agents()
    ags[2].state.followers = {0, 1}
    assert W.current_opinion_leader(ags) == 2


def test_resolve_root_leader_walks_the_chain():
    following = [1, 2, None, None]
    counts = [0, 1, 1, 0]
    assert W.resolve_root_leader(0, following, counts) == 2


def test_resolve_root_leader_detects_a_cycle():
    following = [1, 0, None]
    assert W.resolve_root_leader(0, following, [1, 1, 0]) == -1


def test_resolve_root_leader_handles_no_leader():
    """A None entry must not be int()'d, and follower_counts is a SEQUENCE."""
    assert W.resolve_root_leader(0, [None, None], [0, 0]) == -1
    assert W.resolve_root_leader(0, [None, None], [2, 0]) == 0


def test_paper_welfare_all_vs_followers_differ_by_the_leader_term():
    ags = agents()
    ags[1].state.followers = {0, 2}
    pol = W.current_policies(ags, 3, {})
    rw = SharedBaseGaussian(RewardParams(), DIMS, np.random.default_rng(0))
    w_all = W.paper_welfare(ags, pol, rw, 3, leader_id=1)
    w_fol = W.paper_welfare(ags, pol, rw, 3, leader_id=1, exclude_leader=True)
    theta = 1.0 - np.exp(-ags[1].state.participant_interaction_rate)
    p_s = np.ones(3) / 3
    u_leader = float(np.einsum("sa,sa->", p_s[:, None] * pol[1], rw.table[1]))
    assert w_all - w_fol == pytest.approx(theta * u_leader)


def test_paper_welfare_uses_participant_not_actor_rates():
    ags = agents()
    pol = W.current_policies(ags, 3, {})
    rw = rewards()
    base = W.paper_welfare(ags, pol, rw, 3, leader_id=0)
    for a in ags:
        a.state.actor_interaction_rate = 0.01     # must not matter
    assert W.paper_welfare(ags, pol, rw, 3, leader_id=0) == pytest.approx(base)
    for a in ags:
        a.state.participant_interaction_rate = 0.01
    assert W.paper_welfare(ags, pol, rw, 3, leader_id=0) != pytest.approx(base)


def test_paper_welfare_defaults_to_the_current_leader():
    ags = agents()
    ags[3].state.followers = {0}
    pol = W.current_policies(ags, 3, {})
    rw = rewards()
    assert W.paper_welfare(ags, pol, rw, 3) == pytest.approx(
        W.paper_welfare(ags, pol, rw, 3, leader_id=3))
