"""
Read-only metrics: true reputation, paper welfare (Section 2.2.7 / Definition 5.2).

Design rules:
  * Nothing here mutates. Every function is a pure read over (agents, rewards).
  * No dependency on system.py. These take the pieces they need as arguments so
    the module sits below system in the import graph.
  * `policies` is passed in rather than pulled from agents, because every function
    here needs the SAME role-consistent policy per agent and recomputing it three
    times was a large part of the original's cost.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from model.agent import Agent, AgentRole
from model.rewards import RewardModel, state_probabilities


def current_policies(agents: Sequence[Agent],
                     leader_weights: dict[int, np.ndarray]) -> np.ndarray:
    """
    (N, S, A) array of each agent's role-consistent policy.

    Computed once and shared by every metric below. The original called
    get_current_policy inside triple-nested loops (889, 897, 1848, 1881) —
    N*S*A softmaxes recomputed per metric per tracked step.

    `leader_weights` supplies w_k(t) for REPUTATION agents, resolved by the caller
    (agents no longer hold a system reference).
    """
    W = np.stack([_effective_weights(a, leader_weights) for a in agents])
    logits = W - W.max(axis=2, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=2, keepdims=True)

def _effective_weights(agent: Agent, leader_weights: dict[int, np.ndarray]):
    st = agent.state
    if st.role is AgentRole.REPUTATION and st.following is not None:
        w = leader_weights.get(agent.agent_id)
        if w is not None:
            return w
        return st.weights_pu
    if st.role is AgentRole.STATUS:
        return st.weights_status
    return st.weights_pu


def expected_observer_utilities(policies: np.ndarray, rewards: RewardModel) -> np.ndarray:
    """
    E[u_i(s, x_k)] for observer i under agent k's behaviour, averaged over states.
    Returns (N, N) with observers on rows, sources on columns.
    """
    num_states = policies.shape[1]
    return np.einsum('isa,ksa -> ik', rewards.table, policies) / float(num_states)
    


@dataclass(frozen=True)
class TrueReputation:
    """Replaces the Dict[str, object] returned at 929-939."""
    expected_utilities: np.ndarray        # (N, N)
    theta_mu: np.ndarray                  # (N,)
    sum_expected_utility_others: np.ndarray  # (N,)
    true_reputation: np.ndarray           # (N,)
    true_rank: np.ndarray                 # (N,) int, 1-based
    top_value: float
    exact_top_mask: np.ndarray            # (N,) bool
    near_top_mask: np.ndarray             # (N,) bool
    unique_true_top_agent: int            # -1 when tied


def true_reputation(agents: Sequence[Agent], policies: np.ndarray,
                    rewards: RewardModel) -> TrueReputation:
    """
    R_k = theta(mu_{a,k}) * sum_{i != k} E[u_i(s, x_k)].

    Ranking is by (-value, id) so ties break deterministically by agent id (915).
    `unique_true_top_agent` is -1 unless exactly one agent attains the max within
    1e-12 — keep both tolerances (922-925), they are read separately by diagnostics.
    """
    num_agents = len(agents)
    expected_utilities = expected_observer_utilities(policies, rewards)
    theta_mu = 1.0 - np.exp(
        -np.array([float(agent.state.actor_interaction_rate) for agent in agents], dtype=float)
    )
    sum_expected_utility_others = np.sum(expected_utilities, axis=0) - np.diag(expected_utilities)
    true_rep = theta_mu * sum_expected_utility_others

    ranked = sorted(
        range(num_agents),
        key=lambda i: (-float(true_rep[i]), i)
    )

    true_rank = np.zeros(num_agents, dtype=int)
    for pos, agent_id in enumerate(ranked, start=1):
        true_rank[agent_id] = pos

    top_value = float(np.max(true_rep)) if true_rep.size > 0 else 0.0
    exact_tol = 1e-12
    near_tol = 1e-6
    exact_top_mask = np.isclose(true_rep, top_value, atol=exact_tol, rtol=0.0)
    near_top_mask = np.isclose(true_rep, top_value, atol=near_tol, rtol=0.0)
    exact_top_ids = np.where(exact_top_mask)[0].tolist()
    unique_true_top_agents = int(exact_top_ids[0]) if len(exact_top_ids) == 1 else -1

    return TrueReputation(
        expected_utilities=expected_utilities,
        theta_mu=theta_mu,
        sum_expected_utility_others=sum_expected_utility_others,
        true_reputation=true_rep,
        true_rank=true_rank,
        top_value=top_value,
        exact_top_mask=exact_top_mask,
        near_top_mask=near_top_mask,
        unique_true_top_agent=unique_true_top_agents
    )



def resolve_root_leader(agent_id: int, following: Sequence[Optional[int]],
                        follower_counts: Sequence[int]) -> int:
    """
    Walk the follow chain to its root. Returns -1 if a cycle is detected, or if the
    agent neither follows nor is followed.

    Takes plain sequences rather than agents so it can be unit-tested against
    hand-built chains — cycle detection is the kind of thing worth testing directly.
    """
    leader = following[agent_id]
    if leader is None:
        return int(agent_id) if follower_counts[agent_id] > 0 else -1

    seen = {int(agent_id)}
    while leader >= 0 and leader not in seen:
        seen.add(leader)
        next_leader = following[leader]
        if next_leader is None:
            return int(leader)
        leader = int(next_leader)
    return -1


def current_opinion_leader(agents: Sequence[Agent]) -> int:
    """
    argmax over follower counts. Note np.argmax breaks ties by LOWEST index, which
    differs from the (-value, id) rule in true_reputation — preserved as-is (1819-1823).
    """
    follower_counts = [len(a.state.followers) for a in agents]
    if len(follower_counts) == 0:
        return 0
    return int(np.argmax(follower_counts))


def paper_welfare(agents: Sequence[Agent], policies: np.ndarray, rewards: RewardModel,
                  num_states: int, *, leader_id: Optional[int] = None,
                  exclude_leader: bool = False) -> float:
    """
    W(pi) = sum_i theta(mu_{p,i}) U_i(pi),  U_i(pi) = sum_s p(s) sum_x pi(x|s) u_i(s,x)

    with pi the common norm induced by the opinion leader's policy.

      exclude_leader=False -> W_all              (Section 2.2.7, 1825-1857)
      exclude_leader=True  -> W_followers        (Definition 5.2, 1859-1890)

    The two originals are identical apart from the `if i == leader_id: continue`
    at 1874 — one function with a flag, not two copies.

    Vectorised form: with pi = policies[leader_id] of shape (S, A),
        U = einsum('sa,isa->i', p_s[:, None] * pi, rewards.table)
    replaces the triple loop.
    """
    if leader_id is None:
        leader_id = current_opinion_leader(agents)
    pi = policies[leader_id, :, :]
    p_s = state_probabilities(num_states)
    U = np.einsum('sa, isa->i', p_s[:, None] * pi, rewards.table)
    theta = 1.0 - np.exp(-np.array(
        [float(a.state.participant_interaction_rate) for a in agents], dtype=float
    ))
    if exclude_leader:
        theta = theta.copy()
        theta[int(leader_id)] = 0.0
    return float(theta @ U)