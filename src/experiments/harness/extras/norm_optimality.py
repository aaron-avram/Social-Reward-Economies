"""
Norm-optimality plugin.

This block was identical in status_scaling and reputation_status_scaling. It
compares the final leader's greedy deterministic norm against the brute-force
welfare-maximising deterministic norm.

The brute force is over num_actions ** num_states norms, so it is only feasible
for small state spaces; above 50,000 candidates it raises and the plugin records
the sentinel row (is_final_norm_optimal = -1) rather than failing the sweep.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from experiments.harness.plugins import RunContext, RunPlugin

MAX_BRUTEFORCE_NORMS = 50_000

_SENTINEL: Dict[str, Any] = {
    "final_norm": (),
    "best_norm": (),
    "final_norm_welfare_check": float("nan"),
    "best_norm_welfare": float("nan"),
    "welfare_gap_to_best": float("nan"),
    "is_final_norm_optimal": -1,
}


def deterministic_norm_welfare(
    system, norm_actions: Sequence[int], leader_id: int
) -> float:
    """Followers-only paper welfare for the deterministic norm `norm_actions`.

    States are weighted uniformly; the leader is excluded from the sum.
    """
    num_states = system.config.dims.num_states
    p_s = np.ones(num_states, dtype=float) / float(num_states)
    total = 0.0

    for i, agent in enumerate(system.agents):
        if i == int(leader_id):
            continue

        theta_participant = 1.0 - np.exp(-float(agent.state.participant_interaction_rate))

        U_i = 0.0
        for s in range(num_states):
            action = int(norm_actions[s])
            U_i += float(p_s[s]) * float(system.rewards.observer_utility(i, s, action))

        total += theta_participant * U_i

    return float(total)


def leader_greedy_norm(system, leader_id: int) -> Tuple[int, ...]:
    """The final leader's policy collapsed to argmax action per state."""
    leader = system.agents[int(leader_id)]
    actions = []
    for s in range(system.config.dims.num_states):
        pi_s = leader.get_current_policy(s, leader.get_behavior_weights())
        actions.append(int(np.argmax(pi_s)))
    return tuple(actions)


def bruteforce_best_norm(system, leader_id: int) -> Tuple[Tuple[int, ...], float]:
    """Exhaustive search over deterministic norms."""
    num_states = int(system.config.dims.num_states)
    num_actions = int(system.config.dims.num_actions)

    total_norms = num_actions ** num_states
    if total_norms > MAX_BRUTEFORCE_NORMS:
        raise ValueError(
            f"Bruteforce norm search too large: {num_actions}^{num_states} = "
            f"{total_norms}. Use only for small state/action spaces."
        )

    best_norm = None
    best_welfare = -np.inf
    for norm_actions in itertools.product(range(num_actions), repeat=num_states):
        welfare = deterministic_norm_welfare(system, norm_actions, leader_id)
        if welfare > best_welfare:
            best_welfare = welfare
            best_norm = tuple(int(x) for x in norm_actions)

    return best_norm, float(best_welfare)


def final_norm_optimality(system, leader_id: int) -> Dict[str, Any]:
    if leader_id < 0:
        out = dict(_SENTINEL)
        out["is_final_norm_optimal"] = 0  # no leader is a definite non-optimum
        return out

    final_norm = leader_greedy_norm(system, leader_id)
    final_welfare = deterministic_norm_welfare(system, final_norm, leader_id)
    best_norm, best_welfare = bruteforce_best_norm(system, leader_id)

    gap = float(best_welfare - final_welfare)
    is_optimal = int(
        tuple(final_norm) == tuple(best_norm)
        or np.isclose(gap, 0.0, atol=1e-10, rtol=0.0)
    )

    return {
        "final_norm": final_norm,
        "best_norm": best_norm,
        "final_norm_welfare_check": float(final_welfare),
        "best_norm_welfare": float(best_welfare),
        "welfare_gap_to_best": float(gap),
        "is_final_norm_optimal": int(is_optimal),
    }


def norm_to_str(norm: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in norm)


class NormOptimalityPlugin(RunPlugin):
    name = "norm_optimality"
    columns = (
        "final_norm",
        "best_norm",
        "final_norm_welfare_check",
        "best_norm_welfare",
        "welfare_gap_to_best",
        "is_final_norm_optimal",
    )

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        try:
            result = final_norm_optimality(ctx.system, ctx.summary.leader_id)
        except ValueError:
            result = dict(_SENTINEL)

        return {
            "final_norm": norm_to_str(result["final_norm"]) if result["final_norm"] else "",
            "best_norm": norm_to_str(result["best_norm"]) if result["best_norm"] else "",
            "final_norm_welfare_check": float(result["final_norm_welfare_check"]),
            "best_norm_welfare": float(result["best_norm_welfare"]),
            "welfare_gap_to_best": float(result["welfare_gap_to_best"]),
            "is_final_norm_optimal": int(result["is_final_norm_optimal"]),
        }
