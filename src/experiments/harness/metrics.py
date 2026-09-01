"""
Metric computations over a finished simulation.

Every function here appeared verbatim (or near-verbatim) in three or four of the
legacy harnesses. They are pure functions of arrays, which makes them unit
testable in isolation -- the legacy copies were only reachable by running a full
sweep.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def role_to_label(role) -> str:
    if hasattr(role, "value"):
        return str(role.value).lower()
    if hasattr(role, "name"):
        return str(role.name).lower()
    return str(role).lower()


def finalize_results(system) -> Dict:
    """Snapshot terminal state alongside the recorded history, as a plain dict.

    The engine returns a `SimulationResults` dataclass. Flattening it to a dict
    here keeps the metric functions below independent of the engine's record
    type, which is what makes them testable against synthetic arrays.

    The terminal fields are recomputed rather than read from
    `system._finalize()`, because the harness drives `step()` directly instead of
    calling `simulate()` and so never triggers the engine's own finalisation.
    """
    from dataclasses import fields

    results = {f.name: getattr(system.results, f.name) for f in fields(system.results)}

    final_followers = [len(a.state.followers) for a in system.agents]
    results["final_roles"] = [a.state.role for a in system.agents]
    results["final_followers"] = final_followers
    results["opinion_leader"] = (
        int(np.argmax(final_followers)) if max(final_followers, default=0) > 0 else -1
    )
    return results


def leader_series_from_follower_counts(follower_counts: np.ndarray) -> np.ndarray:
    """Per-step leader identity, ties broken by lowest agent id. -1 if no followers."""
    leaders = np.full(shape=(follower_counts.shape[0],), fill_value=-1, dtype=int)
    for t in range(follower_counts.shape[0]):
        row = follower_counts[t]
        m = int(np.max(row))
        if m > 0:
            candidates = np.where(row == m)[0]
            leaders[t] = int(candidates[0])
    return leaders


def leader_series_hysteretic(follower_counts: np.ndarray, margin: int) -> np.ndarray:
    """Leader identity with an incumbency margin.

    The plain series breaks ties by lowest agent id, so two agents trading a tie
    register as a switch every step -- and kappa makes near-ties MORE common
    (more agents clear the c*N status gate at comparable follower counts).
    Retain the incumbent unless a challenger strictly exceeds it by `margin`
    followers. margin=0 reproduces the plain series.
    """
    if margin <= 0:
        return leader_series_from_follower_counts(follower_counts)
    T = follower_counts.shape[0]
    leaders = np.full(shape=(T,), fill_value=-1, dtype=int)
    incumbent = -1
    for t in range(T):
        row = follower_counts[t]
        best = int(np.argmax(row))
        if int(row[best]) <= 0:
            incumbent = -1
        elif incumbent < 0 or int(row[best]) > int(row[incumbent]) + int(margin):
            incumbent = best
        leaders[t] = incumbent
    return leaders


def leader_switches(leader_series: np.ndarray) -> int:
    """Count changes of leader identity, ignoring leaderless steps."""
    non_null = [x for x in leader_series.tolist() if x >= 0]
    if len(non_null) <= 1:
        return 0
    return sum(1 for a, b in zip(non_null[:-1], non_null[1:]) if a != b)


def time_to_threshold(series: np.ndarray, threshold: int) -> int:
    """First 1-indexed timestep at which `series` reaches `threshold`; -1 if never."""
    idx = np.where(series >= threshold)[0]
    return int(idx[0] + 1) if idx.size > 0 else -1


def tail_top_follower_share(
    follower_counts: np.ndarray, tail_window: int, denom: int
) -> float:
    """Mean top-follower count over the tail, normalised by `denom`.

    NOTE: Experiment A's legacy copy normalised inside the mean and clamped the
    denominator with max(1, denom); B/C normalised outside and guarded with an
    early return. The two agree whenever denom > 0, which holds for every
    committed run. `divide_inside` selects the A variant for exact parity.
    """
    if follower_counts.size == 0 or tail_window <= 0 or denom <= 0:
        return 0.0
    tail_window = min(int(tail_window), follower_counts.shape[0])
    tail = follower_counts[-tail_window:]
    return float(np.mean(tail.max(axis=1)) / float(denom))


def tail_top_follower_share_elementwise(
    follower_counts: np.ndarray, tail_window: int, denom: int
) -> float:
    """Experiment A variant: divide before averaging, denominator clamped to >= 1."""
    if follower_counts.size == 0:
        return 0.0
    tail_window = min(int(tail_window), follower_counts.shape[0])
    tail = follower_counts[-tail_window:]
    return float(np.mean(np.max(tail, axis=1) / max(1, denom)))


def tail_status_leader_share(
    role_history: np.ndarray, leader_series: np.ndarray, tail_window: int
) -> float:
    """Fraction of tail steps (with a leader) on which the leader was in STATUS."""
    if role_history.size == 0 or leader_series.size == 0 or tail_window <= 0:
        return 0.0
    tail_window = min(tail_window, leader_series.shape[0])
    start = leader_series.shape[0] - tail_window
    hits = 0
    valid = 0
    for t in range(start, leader_series.shape[0]):
        leader = int(leader_series[t])
        if leader >= 0:
            valid += 1
            if role_to_label(role_history[t, leader]) == "status":
                hits += 1
    return float(hits / valid) if valid > 0 else 0.0


def tail_status_agent_share(role_history: np.ndarray, tail_window: int) -> float:
    """Fraction of (tail step, agent) pairs in STATUS."""
    if role_history.size == 0 or tail_window <= 0:
        return 0.0
    tail_window = min(tail_window, role_history.shape[0])
    tail = role_history[-tail_window:]
    tail_labels = np.vectorize(role_to_label)(tail)
    return float(np.mean(tail_labels == "status"))


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    """Sample mean, sample std (ddof=1), and a normal-approximation 95% half-width.

    The interval is 1.96 * s / sqrt(n), which is the large-sample normal
    approximation, not a t-interval. With the 3-10 seeds these sweeps typically
    use, it understates the true 95% width by roughly 10-25%. Kept as-is for
    parity with published figures; treat the bars as indicative.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size >= 2 else 0.0
    return mean, std, ci95
