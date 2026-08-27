"""
Parity for the read-only metrics: welfare and true reputation.

TOLERANCE: 1e-6, not exact. The benchmark's get_softmax_policy divides by
(sum(exp) + 1e-8), so its policies sum to 0.999999993 rather than 1. The package
normalises properly. Measured effect: 5.8e-9 on expected utilities, 1.1e-8 on
welfare — the epsilon and nothing else. This is a deliberate fix, so the test
pins the size of the difference rather than demanding zero.
"""
import numpy as np
import pytest

from parity.harness import bench_system, load_benchmark, set_bench_rep_state
from model.agent import Agent, AgentRole
from model.config import AlgorithmParams, Dimensions, RewardParams, RewardModelKind
from model.rewards import build_reward_model
from model import welfare as W

bm = load_benchmark()
N, S, A = 6, 3, 2
DIMS = Dimensions(num_agents=N, num_states=S, num_actions=A)


def matched_pair(seed):
    """Benchmark system and package agents with identical weights, rates, roles."""
    g = np.random.default_rng(seed)
    wpu = g.normal(size=(N, S, A)) * 0.5
    wst = g.normal(size=(N, S, A)) * 0.5
    rates_a = g.uniform(0.1, 1.0, size=N)
    rates_p = g.uniform(0.1, 1.0, size=N)

    sysb = bench_system(bm, N)
    params = AlgorithmParams()
    agents = [Agent(i, params, DIMS, np.random.default_rng(50 + i)) for i in range(N)]

    for i in range(N):
        for holder in (sysb.agents[i].state, agents[i].state):
            holder.weights_pu = wpu[i].copy()
            holder.weights_status = wst[i].copy()
            holder.actor_interaction_rate = float(rates_a[i])
            holder.participant_interaction_rate = float(rates_p[i])

    rewards = build_reward_model(
        RewardParams(kind=RewardModelKind.SIMPLE_PREFERRED_ACTION), DIMS,
        np.random.default_rng(0))
    return sysb, agents, rewards


@pytest.mark.parametrize("seed", range(10))
def test_expected_observer_utilities_match(seed):
    """The einsum orientation: rows are observers, columns are sources."""
    sysb, agents, rewards = matched_pair(seed)
    want = sysb._compute_expected_observer_utilities_by_agent()
    got = W.expected_observer_utilities(W.current_policies(agents, {}), rewards)
    assert np.allclose(got, want, atol=1e-6, rtol=0)


@pytest.mark.parametrize("seed", range(10))
def test_true_reputation_matches(seed):
    sysb, agents, rewards = matched_pair(seed)
    want = sysb._compute_true_reputation_vector()
    got = W.true_reputation(agents, W.current_policies(agents, {}), rewards)
    assert np.allclose(got.true_reputation, want["true_reputation"], atol=1e-6, rtol=0)
    assert np.array_equal(got.true_rank, want["true_rank"])
    assert np.allclose(got.theta_mu, want["theta_mu"], atol=1e-12, rtol=0)  # no softmax involved
    assert got.unique_true_top_agent == want["unique_true_top_agent"]


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("leader", range(N))
def test_paper_welfare_all_agents_matches(seed, leader):
    sysb, agents, rewards = matched_pair(seed)
    want = sysb.compute_paper_welfare_all_agents(leader_id=leader)
    got = W.paper_welfare(agents, W.current_policies(agents, {}), rewards, S,
                          leader_id=leader)
    assert got == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("leader", range(N))
def test_paper_welfare_followers_only_matches(seed, leader):
    sysb, agents, rewards = matched_pair(seed)
    want = sysb.compute_paper_welfare_followers_only(leader_id=leader)
    got = W.paper_welfare(agents, W.current_policies(agents, {}), rewards, S,
                          leader_id=leader, exclude_leader=True)
    assert got == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_welfare_matches_with_status_and_reputation_roles(seed):
    """
    Role-consistent policies: STATUS uses weights_status, followers use the
    leader's behaviour weights.

    NOTE each side must be assigned its OWN AgentRole enum. The benchmark defines
    AgentRole separately from model.agent, and members of different Enum classes
    never compare equal — assigning the package's enum to a benchmark agent makes
    its `role == AgentRole.STATUS` check silently False and it falls through to
    weights_pu. (That is exactly the duplicate-enum hazard flagged in the review,
    and it produced a 3.5e-3 welfare discrepancy here before being spotted.)
    """
    sysb, agents, rewards = matched_pair(seed)

    sysb.agents[1].state.role = bm.AgentRole.STATUS
    sysb.agents[2].state.role = bm.AgentRole.REPUTATION
    agents[1].state.role = AgentRole.STATUS
    agents[2].state.role = AgentRole.REPUTATION
    for holder in (sysb.agents, agents):
        holder[2].state.following = 1
        holder[1].state.followers = {2}

    leader_weights = {2: agents[1].state.weights_status}
    want = sysb.compute_paper_welfare_all_agents(leader_id=1)
    got = W.paper_welfare(agents, W.current_policies(agents, leader_weights),
                          rewards, S, leader_id=1)
    assert got == pytest.approx(want, abs=1e-6)


def test_softmax_epsilon_is_the_only_source_of_disagreement():
    """Pin the size of the known difference so a real divergence cannot hide
    inside a loose tolerance."""
    worst = 0.0
    for seed in range(20):
        sysb, agents, rewards = matched_pair(seed)
        pol = W.current_policies(agents, {})
        worst = max(worst, float(np.max(np.abs(
            W.expected_observer_utilities(pol, rewards)
            - sysb._compute_expected_observer_utilities_by_agent()))))
    assert worst < 1e-7, f"disagreement {worst:.2e} exceeds the softmax epsilon"
