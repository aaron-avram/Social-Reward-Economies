"""
Experiment D harness: opinion-leader perturbation and recovery dynamics.

This runner detects convergence, perturbs the current top leader for a fixed
duration, and measures collapse -> normlessness -> re-emergence behavior.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.code_debugged import AgentRole, MultiAgentSystem, SystemConfig  # noqa: E402


@dataclass
class RunRecord:
    mode: str
    gamma: float
    kappa: float
    seed: int
    converged: bool
    t_conv: int
    leader_pre: int
    pre_followers: int
    t_perturb_start: int
    t_perturb_end: int
    drop_min: float
    drop_fraction: float
    time_to_drop: int
    normless_duration: int
    pu_share_peak_during_drop: float
    recovery_time: int
    leader_post_recovery: int
    leader_changed: bool
    stable_recovery: bool
    stable_tail_window: int
    welfare_pre: float
    welfare_drop: float
    welfare_recovered: float
    final_leader: int
    final_leader_changed: bool
    final_top_followers: int
    post_perturb_role_updates_available: int
    max_alt_leader_followers_post: int
    time_to_alt_leader_25pct: int
    time_to_alt_leader_50pct: int
    time_to_alt_leader_75pct: int
    final_share_positive_step1_margin: float
    final_pu_share: float


@dataclass
class AggregateRecord:
    mode: str
    gamma: float
    kappa: float
    n_runs: int
    conv_rate: float
    drop_rate: float
    normless_rate: float
    recovery_rate: float
    stable_recovery_rate: float
    mean_drop_fraction: float
    std_drop_fraction: float
    ci95_drop_fraction: float
    mean_normless_duration: float
    std_normless_duration: float
    ci95_normless_duration: float
    mean_recovery_time: float
    std_recovery_time: float
    ci95_recovery_time: float
    mean_final_top_followers: float
    std_final_top_followers: float
    ci95_final_top_followers: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment D perturbation/recovery harness.")
    parser.add_argument("--mode", choices=["static", "async"], default="static")

    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=2)
    parser.add_argument("--num-steps-max", type=int, default=12000)

    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--B-R", dest="B_R", type=float, default=0.3)
    parser.add_argument("--B-F", dest="B_F", type=float, default=0.15)
    parser.add_argument("--c-threshold", dest="c_threshold", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--selected-seeds",
        type=str,
        default="",
        help="Optional comma-separated explicit seed list. Overrides --seeds/--seed-start when provided.",
    )
    parser.add_argument(
        "--decision-audit-seeds",
        type=str,
        default="",
        help="Optional comma-separated seed list for per-agent decision-audit export.",
    )
    parser.add_argument(
        "--decision-audit-role-update-steps",
        type=str,
        default="",
        help="Optional comma-separated role-update timesteps to retain in decision-audit exports.",
    )

    parser.add_argument("--role-update-s0", type=int, default=0)
    parser.add_argument("--role-update-T-seq", type=str, default="")
    parser.add_argument("--role-update-base-interval", type=int, default=3000)
    parser.add_argument(
        "--fixed-role-update-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use constant role-update epochs T_n = const (Section 7.1.4) when enabled.",
    )
    parser.add_argument("--role-update-epochs", type=str, default="")
    parser.add_argument(
        "--async-role-update-prob",
        type=float,
        default=None,
        help=(
            "Optional per-step Bernoulli probability for role updates in async mode. "
            "If omitted, async uses independent per-agent clocks from role update schedule."
        ),
    )

    parser.add_argument("--perturb-strength", type=float, default=8.0)
    parser.add_argument("--perturb-duration", type=int, default=600)
    parser.add_argument(
        "--perturb-policy-mode",
        choices=["targeted_low_payoff", "force_bad_action"],
        default="targeted_low_payoff",
        help="How to perturb the leader's action policy during the perturbation window.",
    )
    parser.add_argument(
        "--collapse-followers-on-perturb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, dissolve the pre-leader's follower set at perturbation start.",
    )
    parser.add_argument(
        "--reputation-shock-factor",
        type=float,
        default=1.0,
        help="Per-perturbed-step multiplicative shock on s_i(leader, t) (1.0 disables).",
    )
    parser.add_argument("--post-window", type=int, default=2500)

    parser.add_argument(
        "--conv-threshold",
        type=float,
        default=None,
        help="Convergence threshold on top followers. If <=1, treated as ratio of (N-1).",
    )
    parser.add_argument("--conv-hold-steps", type=int, default=200)

    parser.add_argument(
        "--recovery-threshold",
        type=float,
        default=0.9,
        help="Recovery threshold on top followers. If <=1, treated as ratio of (N-1).",
    )
    parser.add_argument("--recovery-hold-steps", type=int, default=150)
    parser.add_argument(
        "--stable-tail-window",
        type=int,
        default=200,
        help="Require strict convergence in the last K steps to count as stable recovery.",
    )

    parser.add_argument(
        "--dominant-threshold",
        type=float,
        default=0.5,
        help="Dominance threshold for normlessness. If <=1, treated as ratio of (N-1).",
    )
    parser.add_argument(
        "--drop-fraction-threshold",
        type=float,
        default=0.5,
        help="Time-to-drop threshold as fraction of pre-followers.",
    )

    parser.add_argument("--tail-window", type=int, default=500)

    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional subdirectory name under --output-dir for this run.",
    )
    parser.add_argument(
        "--auto-run-subdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, write outputs to a unique subdirectory per run.",
    )
    parser.add_argument("--plot-sample-interval", type=int, default=250)

    parser.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    parser.add_argument("--initial-actor-rate", type=float, default=0.7)
    parser.add_argument("--initial-participant-rate", type=float, default=0.7)
    parser.add_argument(
        "--reward-model",
        choices=["simple_preferred_action", "shared_base_gaussian", "shared_good_bad_heterogeneous"],
        default="simple_preferred_action",
    )
    parser.add_argument("--reward-base-mu", type=float, default=0.5)
    parser.add_argument("--reward-base-sigma", type=float, default=0.08)
    parser.add_argument("--reward-agent-sigma", type=float, default=0.1)
    parser.add_argument("--reward-clip-min", type=float, default=0.01)
    parser.add_argument("--reward-clip-max", type=float, default=2.5)
    parser.add_argument("--reward-good-value", type=float, default=1.0)
    parser.add_argument("--reward-bad-value", type=float, default=0.1)
    parser.add_argument("--reward-order-gap", type=float, default=0.02)

    parser.add_argument(
        "--numpy-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vectorized reputation updates in code_debugged.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="perturbation_recovery",
        help="Prefix for generated artifacts.",
    )

    return parser.parse_args()


def parse_role_update_epochs(epoch_text: str) -> List[int]:
    if not epoch_text.strip():
        return []
    parts = [p.strip() for p in epoch_text.split(",") if p.strip()]
    epochs = [int(x) for x in parts]
    return sorted(set(e for e in epochs if e > 0))


def parse_role_update_T_seq(t_text: str) -> List[int]:
    if not t_text.strip():
        return []
    parts = [p.strip() for p in t_text.split(",") if p.strip()]
    seq = [int(x) for x in parts]
    return [t for t in seq if t > 0]


def parse_selected_seeds(seed_text: str) -> List[int]:
    if not str(seed_text).strip():
        return []
    seeds: List[int] = []
    seen = set()
    for part in [p.strip() for p in str(seed_text).split(",") if p.strip()]:
        seed = int(part)
        if seed < 0:
            raise ValueError(f"selected seeds must be non-negative, got {seed}")
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return seeds


def parse_positive_int_list(int_text: str) -> List[int]:
    if not str(int_text).strip():
        return []
    values: List[int] = []
    seen = set()
    for part in [p.strip() for p in str(int_text).split(",") if p.strip()]:
        value = int(part)
        if value <= 0:
            raise ValueError(f"expected positive integers, got {value}")
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _normalize_selected_seeds(selected_seeds: Optional[Sequence[int] | str]) -> List[int]:
    if selected_seeds is None:
        return []
    if isinstance(selected_seeds, str):
        return parse_selected_seeds(selected_seeds)

    normalized: List[int] = []
    seen = set()
    for seed_like in selected_seeds:
        seed = int(seed_like)
        if seed < 0:
            raise ValueError(f"selected seeds must be non-negative, got {seed}")
        if seed not in seen:
            seen.add(seed)
            normalized.append(seed)
    return normalized


def _normalize_positive_int_sequence(values: Optional[Sequence[int] | str]) -> List[int]:
    if values is None:
        return []
    if isinstance(values, str):
        return parse_positive_int_list(values)
    normalized: List[int] = []
    seen = set()
    for value_like in values:
        value = int(value_like)
        if value <= 0:
            raise ValueError(f"expected positive integers, got {value}")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _resolve_seed_list(args: argparse.Namespace) -> List[int]:
    explicit_seeds = _normalize_selected_seeds(getattr(args, "selected_seeds", None))
    if explicit_seeds:
        return explicit_seeds
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))


def _resolve_decision_audit_seed_set(args: argparse.Namespace) -> set[int]:
    return set(_normalize_selected_seeds(getattr(args, "decision_audit_seeds", None)))


def _resolve_decision_audit_steps(args: argparse.Namespace) -> List[int]:
    return parse_positive_int_list(getattr(args, "decision_audit_role_update_steps", ""))


def _interval_seq_from_epochs(s0: int, epochs: Sequence[int]) -> List[int]:
    prev = max(0, int(s0))
    intervals: List[int] = []
    for epoch in sorted(set(int(e) for e in epochs if int(e) > 0)):
        if epoch > prev:
            intervals.append(int(epoch - prev))
            prev = int(epoch)
    return intervals


def _build_async_interval_sequence(args: argparse.Namespace) -> Tuple[List[int], int, str]:
    s0 = max(0, int(args.role_update_s0))
    t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    if t_seq:
        return t_seq, s0, "T_sequence"

    epochs = parse_role_update_epochs(args.role_update_epochs)
    if epochs:
        from_epochs = _interval_seq_from_epochs(s0=s0, epochs=epochs)
        if from_epochs:
            return from_epochs, s0, "epochs"

    return [max(1, int(args.role_update_base_interval))], s0, "base_interval"


def _resolve_threshold(value: Optional[float], n_minus_1: int, default_abs: float) -> int:
    if value is None:
        return int(np.ceil(default_abs))
    if value <= 1.0:
        return int(np.ceil(value * n_minus_1))
    return int(np.ceil(value))


def derive_interval_scaled_windows(role_update_interval: int) -> Dict[str, int]:
    """
    Keep timing windows proportional to the role-update cadence.

    Baseline ratios:
    - perturb_duration: 3.0 intervals
    - conv_hold_steps: 1.2 intervals
    - recovery_hold_steps: 0.8 intervals
    - stable_tail_window: 2.0 intervals
    """
    interval = max(1, int(role_update_interval))
    return {
        "perturb_duration": int(3 * interval),
        "conv_hold_steps": int(np.ceil(1.2 * interval)),
        "recovery_hold_steps": int(np.ceil(0.8 * interval)),
        "stable_tail_window": int(2 * interval),
    }


def detect_first_hold_index(
    series: Sequence[float],
    threshold: float,
    hold_steps: int,
    *,
    start_idx: int = 0,
) -> int:
    streak = 0
    hold_steps = max(1, int(hold_steps))
    start_idx = max(0, int(start_idx))
    for idx in range(start_idx, len(series)):
        if float(series[idx]) >= float(threshold):
            streak += 1
        else:
            streak = 0
        if streak >= hold_steps:
            return idx
    return -1


def longest_true_run(mask: Sequence[bool]) -> int:
    best = 0
    cur = 0
    for m in mask:
        if bool(m):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def compute_normless_duration(
    top_series: Sequence[float],
    dominant_threshold: float,
    *,
    start_idx: int,
    end_idx: int,
) -> int:
    if len(top_series) == 0:
        return 0
    start_idx = max(0, int(start_idx))
    end_idx = min(len(top_series) - 1, int(end_idx))
    if end_idx < start_idx:
        return 0
    arr = np.asarray(top_series, dtype=float)[start_idx:end_idx + 1]
    mask = arr < float(dominant_threshold)
    return longest_true_run(mask)


def compute_alt_leader_stats(follower_counts: Sequence[int], ex_leader_id: int) -> Tuple[int, int]:
    """Return the strongest non-ex-leader follower bloc and its leader id."""
    best_id = -1
    best_followers = 0
    for agent_id, count in enumerate(follower_counts):
        if int(agent_id) == int(ex_leader_id):
            continue
        count_int = int(count)
        if count_int > best_followers:
            best_followers = count_int
            best_id = int(agent_id)
    return best_id, best_followers


def summarize_positive_step1_margins(margins: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(margins, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    positive = arr > 0.0
    share = float(np.mean(positive))
    mean_positive = float(np.mean(arr[positive])) if np.any(positive) else float("nan")
    return share, mean_positive


def detect_first_threshold_timestep(
    series: Sequence[float],
    threshold: float,
    *,
    start_idx: int = 0,
) -> int:
    start_idx = max(0, int(start_idx))
    for idx in range(start_idx, len(series)):
        if float(series[idx]) >= float(threshold):
            return int(idx + 1)
    return -1


def apply_low_payoff_perturbation(agent, strength: float) -> None:
    """
    Force the leader's policy to strongly prefer a non-preferred action.

    This is applied at every perturbed step, so the shock is temporary and
    released automatically when perturbation duration ends.
    """
    strength = float(abs(strength))
    weights_shape = agent.state.weights_pu.shape
    num_states, num_actions = int(weights_shape[0]), int(weights_shape[1])

    forced = np.full((num_states, num_actions), -strength, dtype=float)
    pref = int(agent.preferred_action % max(1, num_actions))
    if num_actions >= 2:
        anti = int((pref + 1) % num_actions)
        forced[:, anti] = strength
        forced[:, pref] = -strength

    agent.state.weights_pu = forced.copy()
    agent.state.weights_status = forced.copy()


def apply_targeted_low_payoff_perturbation(
    system: MultiAgentSystem,
    leader_id: int,
    *,
    strength: float,
    target_ids: Sequence[int],
) -> None:
    """
    Stronger action-only perturbation: in each state, force the leader toward the
    action that minimizes average utility for a chosen target group.

    This stays endogenous because we only change the leader's policy. Followers
    still leave only if the resulting observed utilities drive their Section-7
    follow decision below the PU alternative.
    """
    agent = system.agents[leader_id]
    target_ids = [int(i) for i in target_ids if 0 <= int(i) < len(system.agents) and int(i) != leader_id]
    if not target_ids:
        apply_low_payoff_perturbation(agent, strength)
        return

    strength = float(abs(strength))
    num_states = int(agent.state.weights_pu.shape[0])
    num_actions = int(agent.state.weights_pu.shape[1])
    forced = np.full((num_states, num_actions), -strength, dtype=float)

    for state in range(num_states):
        action_scores = []
        for action in range(num_actions):
            mean_utility = float(
                np.mean([system.compute_observer_utility(observer_id, state, action) for observer_id in target_ids])
            )
            action_scores.append(mean_utility)

        worst_action = int(np.argmin(action_scores))
        forced[state, worst_action] = strength

    agent.state.weights_pu = forced.copy()
    agent.state.weights_status = forced.copy()


def apply_force_bad_action_perturbation(system: MultiAgentSystem, leader_id: int, *, strength: float) -> None:
    """
    Force the leader away from the designated shared good action in every state.

    This perturbation is intended for the shared_good_bad_heterogeneous reward
    model, where each state stores a system-wide good action g_hat(s).
    """
    good_actions = getattr(system, "_shared_good_actions", None)
    if good_actions is None:
        raise ValueError("force_bad_action perturbation requires system._shared_good_actions.")

    agent = system.agents[leader_id]
    strength = float(abs(strength))
    num_states = int(agent.state.weights_pu.shape[0])
    num_actions = int(agent.state.weights_pu.shape[1])

    forced = np.full((num_states, num_actions), strength, dtype=float)
    for state in range(num_states):
        good_action = int(good_actions[state])
        forced[state, good_action] = -strength

    agent.state.weights_pu = forced.copy()
    agent.state.weights_status = forced.copy()


def apply_reputation_shock(system: MultiAgentSystem, leader_id: int, factor: float) -> None:
    """
    Apply a one-time multiplicative reputation shock to the perturbed leader:
    s_i(leader, t) <- factor * s_i(leader, t) for all agents i.
    """
    factor = float(factor)
    if leader_id < 0 or factor >= 1.0:
        return

    for agent in system.agents:
        cur = float(agent.state.reputation_estimates.get(leader_id, 0.0))
        agent.state.reputation_estimates[leader_id] = factor * cur

    # Keep dense and dict representations consistent.
    if getattr(system, "_s_matrix", None) is not None:
        system._s_matrix[:, leader_id] *= factor


def collapse_leader_followership(system: MultiAgentSystem, leader_id: int) -> None:
    """
    Strong perturbation: dissolve the current leader's follower set in one shot.

    Followers revert to personal-utility mode and stop following, after which the
    standard Section-7 dynamics determine whether and how leadership re-emerges.
    """
    if leader_id < 0 or leader_id >= len(system.agents):
        return

    leader = system.agents[leader_id]
    follower_ids = list(leader.state.followers)
    for follower_id in follower_ids:
        follower = system.agents[follower_id]
        follower.state.following = None
        follower.state.role = AgentRole.PERSONAL_UTILITY
        follower.state.was_following = False
    leader.state.followers.clear()


def _step_has_static_role_update(args: argparse.Namespace, step: int) -> bool:
    """Best-effort marker for synchronous role-update epochs used in diagnostics."""
    if str(args.mode) != "static":
        return False

    explicit_epochs = parse_role_update_epochs(str(args.role_update_epochs))
    if explicit_epochs:
        return int(step) in explicit_epochs

    t_seq = parse_role_update_T_seq(str(args.role_update_T_seq))
    if t_seq:
        cursor = max(0, int(args.role_update_s0))
        for interval in t_seq:
            cursor += int(interval)
            if int(step) == cursor:
                return True
        return False

    if bool(args.fixed_role_update_interval):
        interval = max(1, int(args.role_update_base_interval))
        return int(step) % interval == 0

    return False


def _compute_step1_diagnostic_terms(
    system: MultiAgentSystem,
    agent_id: int,
) -> Dict[str, float | int | bool | None]:
    """
    Mirror code_debugged.py Step-1 semantics for a single agent.

    Diagnostics must use the same currently selected non-self target as the
    role-update logic, not a self-inclusive maximum over all reputation entries.
    """
    agent = system.agents[agent_id]
    config = system.config

    in_C = len(agent.state.followers) == 0
    in_R = agent.state.role == AgentRole.REPUTATION
    hysteresis_active = bool(in_C and in_R and float(config.B_F) < float(config.B_R))
    B_i = float(config.B_F) if hysteresis_active else float(config.B_R)

    target_k = agent.state.highest_rep_agent_estimate
    selected_rep_raw = 0.0
    if target_k is not None:
        selected_rep_raw = float(agent.state.reputation_estimates.get(target_k, 0.0))

    selected_rep_weighted = float(config.gamma) * selected_rep_raw
    est_pu = float(agent.state.estimated_reward_pu)
    threshold = max(B_i, est_pu)

    return {
        "target_id": None if target_k is None else int(target_k),
        "hysteresis_active": bool(hysteresis_active),
        "selected_rep_raw": float(selected_rep_raw),
        "selected_rep_weighted": float(selected_rep_weighted),
        "estimated_reward_pu": float(est_pu),
        "effective_threshold": float(B_i),
        "threshold": float(threshold),
        "step1_margin": float(selected_rep_weighted - threshold),
    }


def make_config(args: argparse.Namespace, mode: str) -> SystemConfig:
    role_interval = int(args.role_update_base_interval)
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)

    if mode == "async":
        role_interval = int(args.num_steps_max) + 1_000_000
        role_s0 = 0
        role_t_seq = []
        role_epochs = []

    return SystemConfig(
        num_agents=int(args.num_agents),
        num_states=int(args.num_states),
        num_actions=int(args.num_actions),
        num_time_steps=int(args.num_steps_max),
        M=1.0,
        u_0=0.1,
        gamma=float(args.gamma),
        kappa=float(args.kappa),
        c_threshold=float(args.c_threshold),
        B_R=float(args.B_R),
        B_F=float(args.B_F),
        delta=float(args.delta),
        alpha_pu_base=0.05,
        beta_status_base=0.05,
        eta_v_base=0.1,
        eta_s_base=0.1,
        eta_J_base=0.05,
        role_update_s0=role_s0,
        role_update_T_sequence=role_t_seq,
        role_update_base_interval=role_interval,
        fixed_role_update_interval=bool(args.fixed_role_update_interval),
        role_update_epochs=role_epochs,
        gossip_rate=0.5,
        gossip_alpha=0.5,
        tracking_mode=str(args.tracking_mode),
        use_numpy_fast_path=bool(args.numpy_fast_path),
        initial_actor_interaction_rate=float(args.initial_actor_rate),
        initial_participant_interaction_rate=float(args.initial_participant_rate),
        reward_model=str(args.reward_model),
        reward_base_mu=float(args.reward_base_mu),
        reward_base_sigma=float(args.reward_base_sigma),
        reward_agent_sigma=float(args.reward_agent_sigma),
        reward_clip_min=float(args.reward_clip_min),
        reward_clip_max=float(args.reward_clip_max),
        reward_good_value=float(args.reward_good_value),
        reward_bad_value=float(args.reward_bad_value),
        reward_order_gap=float(args.reward_order_gap),
    )


def _mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = arr.size
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci95


def _format_num_for_name(x: float) -> str:
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:g}".replace(".", "p").replace("-", "m")


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _build_run_subdir_name(args: argparse.Namespace) -> str:
    if str(args.run_label).strip():
        return _slugify(str(args.run_label))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    explicit_seeds = _normalize_selected_seeds(getattr(args, "selected_seeds", None))
    if explicit_seeds:
        seed_suffix = "seedset_" + "_".join(str(seed) for seed in explicit_seeds)
    else:
        seed_suffix = f"seed{int(args.seed_start)}to{int(args.seed_start) + int(args.seeds) - 1}"
    return (
        f"{_slugify(str(args.output_prefix))}"
        f"_{args.mode}"
        f"_g{_format_num_for_name(args.gamma)}"
        f"_k{_format_num_for_name(args.kappa)}"
        f"_N{int(args.num_agents)}"
        f"_S{int(args.num_states)}"
        f"_steps{int(args.num_steps_max)}"
        f"_{seed_suffix}"
        f"_{stamp}"
    )


def aggregate(records: Sequence[RunRecord]) -> List[AggregateRecord]:
    if not records:
        return []

    conv_vals = [1.0 if r.converged else 0.0 for r in records]
    drop_vals = [1.0 if np.isfinite(r.drop_fraction) and r.drop_fraction > 0 else 0.0 for r in records]
    norm_vals = [1.0 if r.normless_duration > 0 else 0.0 for r in records]
    recov_vals = [1.0 if r.recovery_time > 0 else 0.0 for r in records]
    stable_vals = [1.0 if r.stable_recovery else 0.0 for r in records]

    drop_mean, drop_std, drop_ci = _mean_std_ci([r.drop_fraction for r in records])
    norm_mean, norm_std, norm_ci = _mean_std_ci([float(r.normless_duration) for r in records])
    recov_time_mean, recov_time_std, recov_time_ci = _mean_std_ci(
        [float(r.recovery_time) for r in records if r.recovery_time > 0]
    )
    top_mean, top_std, top_ci = _mean_std_ci([float(r.final_top_followers) for r in records])

    first = records[0]
    return [
        AggregateRecord(
            mode=first.mode,
            gamma=first.gamma,
            kappa=first.kappa,
            n_runs=len(records),
            conv_rate=float(np.mean(conv_vals)),
            drop_rate=float(np.mean(drop_vals)),
            normless_rate=float(np.mean(norm_vals)),
            recovery_rate=float(np.mean(recov_vals)),
            stable_recovery_rate=float(np.mean(stable_vals)),
            mean_drop_fraction=drop_mean,
            std_drop_fraction=drop_std,
            ci95_drop_fraction=drop_ci,
            mean_normless_duration=norm_mean,
            std_normless_duration=norm_std,
            ci95_normless_duration=norm_ci,
            mean_recovery_time=recov_time_mean,
            std_recovery_time=recov_time_std,
            ci95_recovery_time=recov_time_ci,
            mean_final_top_followers=top_mean,
            std_final_top_followers=top_std,
            ci95_final_top_followers=top_ci,
        )
    ]


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_follower_count_map(counts: Counter[int]) -> str:
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
    return ";".join(f"{int(leader)}:{int(count)}" for leader, count in ordered)


def summarize_decision_audit_rows(
    audit_rows: Sequence[Dict[str, object]],
    *,
    leader_pre: int,
) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in audit_rows:
        grouped.setdefault(int(row["t"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for step in sorted(grouped):
        rows = grouped[step]
        following_counts: Counter[int] = Counter(
            int(row["following_after"]) for row in rows if int(row["following_after"]) >= 0
        )
        highest_pre_hits = sum(int(int(row["highest_rep_agent_estimate"]) == int(leader_pre)) for row in rows)
        finite_rep_to_preleader = [
            float(row["rep_to_preleader_raw"])
            for row in rows
            if np.isfinite(float(row["rep_to_preleader_raw"]))
        ]
        top2 = following_counts.most_common(2)
        summary_rows.append(
            {
                "step": int(step),
                "agent_rows": int(len(rows)),
                "leader_pre": int(leader_pre),
                "preleader_follower_count": int(following_counts.get(int(leader_pre), 0)) if leader_pre >= 0 else 0,
                "top_leader_id": int(top2[0][0]) if top2 else -1,
                "top_leader_followers": int(top2[0][1]) if top2 else 0,
                "second_leader_id": int(top2[1][0]) if len(top2) > 1 else -1,
                "second_leader_followers": int(top2[1][1]) if len(top2) > 1 else 0,
                "follower_count_by_leader": _format_follower_count_map(following_counts),
                "share_highest_rep_is_preleader": float(highest_pre_hits / max(1, len(rows))),
                "mean_selected_reputation_raw": float(np.mean([float(r["selected_reputation_raw"]) for r in rows])),
                "mean_rep_to_preleader_raw": float(np.mean(finite_rep_to_preleader)) if finite_rep_to_preleader else float("nan"),
                "mean_estimated_reward_pu": float(np.mean([float(r["estimated_reward_pu"]) for r in rows])),
                "share_step1_condition_met": float(
                    np.mean([1.0 if bool(r["step1_condition_met"]) else 0.0 for r in rows])
                ),
            }
        )
    return summary_rows


def _downsample(x: np.ndarray, y: np.ndarray, sample_interval: int) -> Tuple[np.ndarray, np.ndarray]:
    sample_interval = max(1, int(sample_interval))
    if sample_interval <= 1:
        return x, y
    mask = (x % sample_interval) == 0
    if not np.any(mask):
        mask[-1] = True
    return x[mask], y[mask]


def _plot_seed_trajectory(
    *,
    output_file: Path,
    sample_interval: int,
    top_series: np.ndarray,
    ex_leader_series: np.ndarray,
    pu_share_series: np.ndarray,
    welfare_series: np.ndarray,
    record: RunRecord,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, len(top_series) + 1, dtype=int)
    x_top, y_top = _downsample(x, top_series, sample_interval)
    _, y_ex = _downsample(x, ex_leader_series, sample_interval)
    _, y_pu = _downsample(x, pu_share_series, sample_interval)
    _, y_w = _downsample(x, welfare_series, sample_interval)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax0 = axes[0]
    ax0.plot(x_top, y_top, label="Top followers", linewidth=2.0, color="tab:blue")
    if np.any(np.isfinite(y_ex)):
        ax0.plot(x_top, y_ex, label="Ex-leader followers", linewidth=1.7, color="tab:red", alpha=0.9)
    ax0.set_ylabel("Followers")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1 = axes[1]
    ax1.plot(x_top, y_pu, label="PU share", linewidth=1.8, color="tab:orange")
    ax1.set_ylabel("PU share")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = axes[2]
    ax2.plot(x_top, y_w, label="Social welfare", linewidth=1.8, color="tab:green")
    ax2.set_ylabel("Welfare")
    ax2.set_xlabel("Timestep")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    for ax in axes:
        if record.t_conv > 0:
            ax.axvline(record.t_conv, color="gray", linestyle="--", linewidth=1.2, label="Convergence")
        if record.t_perturb_start > 0:
            ax.axvline(record.t_perturb_start, color="red", linestyle="--", linewidth=1.4, label="Perturb start")
        if record.t_perturb_end > 0:
            ax.axvline(record.t_perturb_end, color="red", linestyle=":", linewidth=1.2, label="Perturb end")
        if record.recovery_time > 0:
            ax.axvline(record.recovery_time, color="purple", linestyle="-.", linewidth=1.2, label="Recovery")

    fig.suptitle(
        f"Perturbation Recovery (seed={record.seed}, mode={record.mode}, gamma={record.gamma:g}, kappa={record.kappa:g})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _finalize_async_step(
    *,
    args: argparse.Namespace,
    system: MultiAgentSystem,
    role_timers: Optional[np.ndarray],
    interval_seq: Optional[List[int]],
    interval_indices: Optional[np.ndarray],
) -> None:
    if args.mode != "async":
        return

    if args.async_role_update_prob is None:
        assert role_timers is not None
        assert interval_seq is not None
        assert interval_indices is not None

        role_timers -= 1
        update_ids = np.where(role_timers <= 0)[0]
        if update_ids.size > 0:
            update_list = update_ids.tolist()
            system._update_roles_sequential(update_list)

            if len(interval_seq) == 1:
                role_timers[update_ids] += int(interval_seq[0])
            else:
                for agent_id in update_list:
                    idx = int(interval_indices[agent_id])
                    next_interval = int(interval_seq[idx if idx < len(interval_seq) else -1])
                    role_timers[agent_id] += next_interval
                    if idx < len(interval_seq) - 1:
                        interval_indices[agent_id] = idx + 1
            system.role_update_epoch += 1
    else:
        p = float(args.async_role_update_prob)
        update_mask = np.random.random(args.num_agents) < p
        update_ids = np.where(update_mask)[0]
        if update_ids.size > 0:
            system._update_roles_sequential(update_ids.tolist())
            system.role_update_epoch += 1


def run_single(args: argparse.Namespace, seed: int) -> Tuple[RunRecord, Dict[str, object]]:
    np.random.seed(int(seed))
    config = make_config(args, mode=args.mode)
    system = MultiAgentSystem(config)
    decision_audit_enabled = int(seed) in _resolve_decision_audit_seed_set(args)
    decision_audit_steps = set(_resolve_decision_audit_steps(args))
    if decision_audit_enabled:
        system.enable_async_decision_audit()
        system.set_decision_audit_preleader(None)

    n = int(args.num_agents)
    n_minus_1 = max(1, n - 1)
    conv_threshold = _resolve_threshold(args.conv_threshold, n_minus_1, default_abs=n_minus_1)
    recovery_threshold = _resolve_threshold(args.recovery_threshold, n_minus_1, default_abs=0.9 * n_minus_1)
    dominant_threshold = _resolve_threshold(args.dominant_threshold, n_minus_1, default_abs=0.5 * n_minus_1)

    conv_hold = max(1, int(args.conv_hold_steps))
    recovery_hold = max(1, int(args.recovery_hold_steps))
    stable_tail_window = max(1, int(args.stable_tail_window))
    perturb_duration = max(1, int(args.perturb_duration))
    post_window = max(1, int(args.post_window))

    role_timers = None
    interval_seq = None
    interval_indices = None
    if args.mode == "async" and args.async_role_update_prob is None:
        interval_seq, async_s0, _ = _build_async_interval_sequence(args)
        first_interval = int(interval_seq[0])
        role_timers = np.random.randint(1, first_interval + 1, size=n, dtype=int)
        if async_s0 > 0:
            role_timers = role_timers + async_s0
        interval_indices = np.zeros(n, dtype=int)

    top_series: List[float] = []
    leader_series: List[int] = []
    alt_leader_series: List[int] = []
    alt_leader_followers_series: List[float] = []
    ex_leader_followers_series: List[float] = []
    pu_share_series: List[float] = []
    welfare_series: List[float] = []
    share_positive_step1_margin_series: List[float] = []
    mean_positive_step1_margin_series: List[float] = []
    role_updates_since_perturb_end_series: List[int] = []
    follower_rows: List[List[int]] = []

    t_conv = -1
    leader_pre = -1
    pre_followers = 0
    t_perturb_start = -1
    t_perturb_end = -1
    leader_post_recovery = -1
    recovery_time = -1
    perturb_target_ids: List[int] = []
    exit_diagnostics: List[Dict[str, object]] = []

    conv_streak = 0
    recovery_streak = 0
    collapse_applied = False
    role_updates_since_perturb_end = 0
    for _ in range(int(args.num_steps_max)):
        next_step = int(system.time_step) + 1
        prev_role_update_epoch = int(system.role_update_epoch)

        if leader_pre >= 0 and t_perturb_start > 0 and t_perturb_start <= next_step <= t_perturb_end:
            if not perturb_target_ids:
                perturb_target_ids = sorted(int(i) for i in system.agents[leader_pre].state.followers)
                if not perturb_target_ids:
                    perturb_target_ids = [i for i in range(n) if i != leader_pre]
            if bool(args.collapse_followers_on_perturb) and not collapse_applied and next_step == t_perturb_start:
                collapse_leader_followership(system, leader_pre)
                collapse_applied = True
            if float(args.reputation_shock_factor) < 1.0:
                apply_reputation_shock(system, leader_pre, float(args.reputation_shock_factor))
            if str(args.perturb_policy_mode) == "force_bad_action":
                apply_force_bad_action_perturbation(
                    system,
                    leader_pre,
                    strength=float(args.perturb_strength),
                )
            else:
                apply_targeted_low_payoff_perturbation(
                    system,
                    leader_pre,
                    strength=float(args.perturb_strength),
                    target_ids=perturb_target_ids,
                )

        with redirect_stdout(io.StringIO()):
            system.step()
        _finalize_async_step(
            args=args,
            system=system,
            role_timers=role_timers,
            interval_seq=interval_seq,
            interval_indices=interval_indices,
        )
        role_update_triggered = int(system.role_update_epoch) > prev_role_update_epoch

        followers = [len(a.state.followers) for a in system.agents]
        follower_rows.append(followers)

        top_followers = int(max(followers)) if followers else 0
        leader = int(np.argmax(followers)) if top_followers > 0 else -1
        pu_share = float(sum(1 for a in system.agents if a.state.role == AgentRole.PERSONAL_UTILITY) / n)
        welfare = float(system.results["social_welfare"][-1])
        alt_leader_id, alt_leader_followers = compute_alt_leader_stats(followers, leader_pre if leader_pre >= 0 else -1)

        top_series.append(float(top_followers))
        leader_series.append(leader)
        alt_leader_series.append(int(alt_leader_id))
        alt_leader_followers_series.append(float(alt_leader_followers))
        pu_share_series.append(pu_share)
        welfare_series.append(welfare)

        if t_conv < 0:
            if top_followers >= conv_threshold:
                conv_streak += 1
            else:
                conv_streak = 0

            if conv_streak >= conv_hold:
                t_conv = int(system.time_step)
                leader_pre = leader
                system.set_decision_audit_preleader(leader_pre)
                pre_followers = top_followers
                if t_conv < int(args.num_steps_max):
                    t_perturb_start = t_conv + 1
                    t_perturb_end = min(int(args.num_steps_max), t_perturb_start + perturb_duration - 1)

        if leader_pre >= 0:
            ex_leader_followers_series.append(float(followers[leader_pre]))
        else:
            ex_leader_followers_series.append(float("nan"))

        if t_perturb_end > 0 and int(system.time_step) >= (t_perturb_end + 1):
            if role_update_triggered:
                role_updates_since_perturb_end += 1
            if top_followers >= recovery_threshold:
                recovery_streak += 1
            else:
                recovery_streak = 0

            if recovery_time < 0 and recovery_streak >= recovery_hold:
                recovery_time = int(system.time_step)
                leader_post_recovery = leader

        if leader_pre >= 0 and t_perturb_start > 0:
            diag_end = min(int(args.num_steps_max), int(t_perturb_end) + int(post_window))
            if t_perturb_start <= int(system.time_step) <= diag_end:
                tracked_ids = perturb_target_ids or [i for i in range(n) if i != leader_pre]
                margins = []
                thresholds = []
                rep_to_leader = []
                gamma_max_reps = []
                highest_pre = 0
                follow_pre = 0
                pu_count = 0
                rep_other_count = 0
                target_count = 0

                for agent_id in tracked_ids:
                    agent = system.agents[agent_id]
                    if agent_id == leader_pre:
                        continue
                    target_count += 1

                    step1_terms = _compute_step1_diagnostic_terms(system, agent_id)

                    thresholds.append(float(step1_terms["threshold"]))
                    margins.append(float(step1_terms["step1_margin"]))
                    rep_to_leader.append(float(agent.state.reputation_estimates.get(leader_pre, 0.0)))
                    gamma_max_reps.append(float(step1_terms["selected_rep_weighted"]))
                    highest_pre += int(step1_terms["target_id"] == leader_pre)
                    follow_pre += int(agent.state.following == leader_pre)
                    pu_count += int(agent.state.role == AgentRole.PERSONAL_UTILITY)
                    rep_other_count += int(
                        agent.state.role == AgentRole.REPUTATION and agent.state.following is not None and agent.state.following != leader_pre
                    )

                share_positive_step1_margin, mean_positive_step1_margin = summarize_positive_step1_margins(margins)
                share_positive_step1_margin_series.append(share_positive_step1_margin)
                mean_positive_step1_margin_series.append(mean_positive_step1_margin)
                role_updates_since_perturb_end_series.append(int(role_updates_since_perturb_end))

                exit_diagnostics.append(
                    {
                        "step": int(system.time_step),
                        "role_update_step": int(role_update_triggered),
                        "leader_pre": int(leader_pre),
                        "leader_pre_followers": int(followers[leader_pre]),
                        "top_followers": int(top_followers),
                        "current_leader": int(leader),
                        "alt_leader_id": int(alt_leader_id),
                        "largest_alt_leader_followers": int(alt_leader_followers),
                        "pu_share": float(pu_share),
                        "tracked_targets": int(target_count),
                        "targets_following_preleader": int(follow_pre),
                        "targets_in_pu": int(pu_count),
                        "targets_in_rep_elsewhere": int(rep_other_count),
                        "mean_gamma_selected_rep": float(np.mean(gamma_max_reps)) if gamma_max_reps else float("nan"),
                        "mean_rep_to_preleader": float(np.mean(rep_to_leader)) if rep_to_leader else float("nan"),
                        "mean_estimated_reward_pu": float(
                            np.mean([system.agents[i].state.estimated_reward_pu for i in tracked_ids if i != leader_pre])
                        ) if target_count > 0 else float("nan"),
                        "mean_threshold": float(np.mean(thresholds)) if thresholds else float("nan"),
                        "mean_step1_margin": float(np.mean(margins)) if margins else float("nan"),
                        "min_step1_margin": float(np.min(margins)) if margins else float("nan"),
                        "share_positive_step1_margin": float(share_positive_step1_margin),
                        "mean_positive_step1_margin": float(mean_positive_step1_margin),
                        "mean_highest_rep_is_preleader": float(highest_pre / target_count) if target_count > 0 else float("nan"),
                        "num_role_updates_since_perturb_end": int(role_updates_since_perturb_end),
                    }
                )

    top_arr = np.array(top_series, dtype=float)
    leader_arr = np.array(leader_series, dtype=int)
    alt_leader_arr = np.array(alt_leader_series, dtype=int)
    alt_follower_arr = np.array(alt_leader_followers_series, dtype=float)
    ex_arr = np.array(ex_leader_followers_series, dtype=float)
    pu_arr = np.array(pu_share_series, dtype=float)
    welfare_arr = np.array(welfare_series, dtype=float)

    final_followers = follower_rows[-1] if follower_rows else [0] * n
    final_top_followers = int(max(final_followers)) if final_followers else 0
    final_leader = int(np.argmax(final_followers)) if final_top_followers > 0 else -1
    stable_tail = top_arr[-min(stable_tail_window, len(top_arr)):] if len(top_arr) > 0 else np.array([], dtype=float)
    stable_recovery = bool(stable_tail.size > 0 and np.all(stable_tail >= conv_threshold))

    drop_min = float("nan")
    drop_fraction = float("nan")
    time_to_drop = -1
    normless_duration = 0
    pu_share_peak_during_drop = float("nan")
    welfare_pre = float("nan")
    welfare_drop = float("nan")
    welfare_recovered = float("nan")
    post_perturb_role_updates_available = 0
    max_alt_leader_followers_post = 0
    time_to_alt_leader_25pct = -1
    time_to_alt_leader_50pct = -1
    time_to_alt_leader_75pct = -1
    final_share_positive_step1_margin = float("nan")
    final_pu_share = float(pu_arr[-1]) if pu_arr.size > 0 else float("nan")

    if t_perturb_start > 0 and leader_pre >= 0:
        start_idx = t_perturb_start - 1
        end_idx = len(top_arr) - 1

        ex_window = ex_arr[start_idx:end_idx + 1]
        valid_ex_window = ex_window[np.isfinite(ex_window)]
        if valid_ex_window.size > 0:
            drop_min = float(np.min(valid_ex_window))
            if pre_followers > 0:
                drop_fraction = float((pre_followers - drop_min) / pre_followers)
                target = float(pre_followers) * float(args.drop_fraction_threshold)
                below = np.where(valid_ex_window <= target)[0]
                if below.size > 0:
                    time_to_drop = int(t_perturb_start + int(below[0]))

        normless_duration = compute_normless_duration(
            top_arr,
            dominant_threshold,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        pu_share_peak_during_drop = float(np.max(pu_arr[start_idx:end_idx + 1]))

        tail = max(1, int(args.tail_window))
        pre_end = max(0, start_idx - 1)
        pre_start = max(0, pre_end - tail + 1)
        if pre_end >= pre_start:
            welfare_pre = float(np.mean(welfare_arr[pre_start:pre_end + 1]))

        drop_end = max(start_idx, min(len(welfare_arr) - 1, t_perturb_end - 1)) if t_perturb_end > 0 else end_idx
        welfare_drop = float(np.mean(welfare_arr[start_idx:drop_end + 1]))

        post_start_idx = min(len(alt_follower_arr), max(0, int(t_perturb_end)))
        post_alt_window = alt_follower_arr[post_start_idx:end_idx + 1]
        if post_alt_window.size > 0:
            max_alt_leader_followers_post = int(np.max(post_alt_window))
            time_to_alt_leader_25pct = detect_first_threshold_timestep(
                alt_follower_arr,
                threshold=float(np.ceil(0.25 * n_minus_1)),
                start_idx=post_start_idx,
            )
            time_to_alt_leader_50pct = detect_first_threshold_timestep(
                alt_follower_arr,
                threshold=float(np.ceil(0.50 * n_minus_1)),
                start_idx=post_start_idx,
            )
            time_to_alt_leader_75pct = detect_first_threshold_timestep(
                alt_follower_arr,
                threshold=float(np.ceil(0.75 * n_minus_1)),
                start_idx=post_start_idx,
            )

        post_perturb_role_updates_available = int(role_updates_since_perturb_end)
        if share_positive_step1_margin_series:
            final_share_positive_step1_margin = float(share_positive_step1_margin_series[-1])

    if recovery_time > 0:
        rec_idx = recovery_time - 1
        rec_end = min(len(welfare_arr) - 1, rec_idx + max(1, int(args.tail_window)) - 1)
        welfare_recovered = float(np.mean(welfare_arr[rec_idx:rec_end + 1]))

    leader_changed = bool(recovery_time > 0 and leader_post_recovery >= 0 and leader_pre >= 0 and leader_post_recovery != leader_pre)
    final_leader_changed = bool(leader_pre >= 0 and final_leader >= 0 and final_leader != leader_pre)

    record = RunRecord(
        mode=str(args.mode),
        gamma=float(args.gamma),
        kappa=float(args.kappa),
        seed=int(seed),
        converged=bool(t_conv > 0),
        t_conv=int(t_conv),
        leader_pre=int(leader_pre),
        pre_followers=int(pre_followers),
        t_perturb_start=int(t_perturb_start),
        t_perturb_end=int(t_perturb_end),
        drop_min=float(drop_min),
        drop_fraction=float(drop_fraction),
        time_to_drop=int(time_to_drop),
        normless_duration=int(normless_duration),
        pu_share_peak_during_drop=float(pu_share_peak_during_drop),
        recovery_time=int(recovery_time),
        leader_post_recovery=int(leader_post_recovery),
        leader_changed=leader_changed,
        stable_recovery=stable_recovery,
        stable_tail_window=int(min(stable_tail_window, len(top_arr))),
        welfare_pre=float(welfare_pre),
        welfare_drop=float(welfare_drop),
        welfare_recovered=float(welfare_recovered),
        final_leader=int(final_leader),
        final_leader_changed=final_leader_changed,
        final_top_followers=int(final_top_followers),
        post_perturb_role_updates_available=int(post_perturb_role_updates_available),
        max_alt_leader_followers_post=int(max_alt_leader_followers_post),
        time_to_alt_leader_25pct=int(time_to_alt_leader_25pct),
        time_to_alt_leader_50pct=int(time_to_alt_leader_50pct),
        time_to_alt_leader_75pct=int(time_to_alt_leader_75pct),
        final_share_positive_step1_margin=float(final_share_positive_step1_margin),
        final_pu_share=float(final_pu_share),
    )

    details = {
        "top_series": top_arr,
        "leader_series": leader_arr,
        "alt_leader_series": alt_leader_arr,
        "alt_leader_followers_series": alt_follower_arr,
        "ex_leader_followers_series": ex_arr,
        "pu_share_series": pu_arr,
        "welfare_series": welfare_arr,
        "follower_rows": follower_rows,
        "perturb_target_ids": list(perturb_target_ids),
        "exit_diagnostics": exit_diagnostics,
        "share_positive_step1_margin_series": np.array(share_positive_step1_margin_series, dtype=float),
        "mean_positive_step1_margin_series": np.array(mean_positive_step1_margin_series, dtype=float),
        "role_updates_since_perturb_end_series": np.array(role_updates_since_perturb_end_series, dtype=int),
        "conv_threshold": conv_threshold,
        "recovery_threshold": recovery_threshold,
        "dominant_threshold": dominant_threshold,
    }
    if decision_audit_enabled:
        audit_rows = system.get_async_decision_audit_rows()
        if decision_audit_steps:
            audit_rows = [row for row in audit_rows if int(row["t"]) in decision_audit_steps]
        details["decision_audit_rows"] = audit_rows
        details["decision_audit_checkpoint_summary_rows"] = summarize_decision_audit_rows(
            audit_rows,
            leader_pre=int(leader_pre),
        )
    return record, details


def run_experiment(
    *,
    mode: str = "static",
    num_agents: int = 8,
    num_states: int = 3,
    num_actions: int = 2,
    num_steps_max: int = 12000,
    gamma: float = 2.0,
    kappa: float = 2.0,
    B_R: float = 0.3,
    B_F: float = 0.15,
    c_threshold: float = 0.1,
    delta: float = 0.15,
    seeds: int = 10,
    seed_start: int = 0,
    selected_seeds: Optional[Sequence[int] | str] = None,
    role_update_s0: int = 0,
    role_update_T_seq: str = "",
    role_update_base_interval: int = 3000,
    fixed_role_update_interval: bool = True,
    role_update_epochs: str = "",
    async_role_update_prob: Optional[float] = None,
    perturb_strength: float = 8.0,
    perturb_duration: int = 600,
    perturb_policy_mode: str = "targeted_low_payoff",
    collapse_followers_on_perturb: bool = False,
    reputation_shock_factor: float = 1.0,
    post_window: int = 2500,
    conv_threshold: Optional[float] = None,
    conv_hold_steps: int = 200,
    recovery_threshold: float = 0.9,
    recovery_hold_steps: int = 150,
    stable_tail_window: int = 200,
    dominant_threshold: float = 0.5,
    drop_fraction_threshold: float = 0.5,
    tail_window: int = 500,
    output_dir: str = str(Path(__file__).resolve().parent / "outputs"),
    run_label: str = "",
    auto_run_subdir: bool = True,
    plot_sample_interval: int = 250,
    tracking_mode: str = "light",
    initial_actor_rate: float = 0.7,
    initial_participant_rate: float = 0.7,
    reward_model: str = "simple_preferred_action",
    reward_base_mu: float = 0.5,
    reward_base_sigma: float = 0.08,
    reward_agent_sigma: float = 0.1,
    reward_clip_min: float = 0.01,
    reward_clip_max: float = 2.5,
    reward_good_value: float = 1.0,
    reward_bad_value: float = 0.1,
    reward_order_gap: float = 0.02,
    decision_audit_seeds: Optional[Sequence[int] | str] = None,
    decision_audit_role_update_steps: Optional[Sequence[int] | str] = None,
    numpy_fast_path: bool = True,
    output_prefix: str = "perturbation_recovery",
) -> Dict[str, object]:
    args = argparse.Namespace(
        mode=mode,
        num_agents=num_agents,
        num_states=num_states,
        num_actions=num_actions,
        num_steps_max=num_steps_max,
        gamma=gamma,
        kappa=kappa,
        B_R=B_R,
        B_F=B_F,
        c_threshold=c_threshold,
        delta=delta,
        seeds=seeds,
        seed_start=seed_start,
        selected_seeds=_normalize_selected_seeds(selected_seeds),
        decision_audit_seeds=_normalize_selected_seeds(decision_audit_seeds),
        decision_audit_role_update_steps=",".join(
            str(step) for step in _normalize_positive_int_sequence(decision_audit_role_update_steps)
        ),
        role_update_s0=role_update_s0,
        role_update_T_seq=role_update_T_seq,
        role_update_base_interval=role_update_base_interval,
        fixed_role_update_interval=fixed_role_update_interval,
        role_update_epochs=role_update_epochs,
        async_role_update_prob=async_role_update_prob,
        perturb_strength=perturb_strength,
        perturb_duration=perturb_duration,
        perturb_policy_mode=perturb_policy_mode,
        collapse_followers_on_perturb=collapse_followers_on_perturb,
        reputation_shock_factor=reputation_shock_factor,
        post_window=post_window,
        conv_threshold=conv_threshold,
        conv_hold_steps=conv_hold_steps,
        recovery_threshold=recovery_threshold,
        recovery_hold_steps=recovery_hold_steps,
        stable_tail_window=stable_tail_window,
        dominant_threshold=dominant_threshold,
        drop_fraction_threshold=drop_fraction_threshold,
        tail_window=tail_window,
        output_dir=output_dir,
        run_label=run_label,
        auto_run_subdir=auto_run_subdir,
        plot_sample_interval=plot_sample_interval,
        tracking_mode=tracking_mode,
        initial_actor_rate=initial_actor_rate,
        initial_participant_rate=initial_participant_rate,
        reward_model=reward_model,
        reward_base_mu=reward_base_mu,
        reward_base_sigma=reward_base_sigma,
        reward_agent_sigma=reward_agent_sigma,
        reward_clip_min=reward_clip_min,
        reward_clip_max=reward_clip_max,
        reward_good_value=reward_good_value,
        reward_bad_value=reward_bad_value,
        reward_order_gap=reward_order_gap,
        numpy_fast_path=numpy_fast_path,
        output_prefix=output_prefix,
    )

    output_root = Path(args.output_dir)
    if bool(args.auto_run_subdir):
        out_dir = output_root / _build_run_subdir_name(args)
    else:
        out_dir = output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records: List[RunRecord] = []
    details_by_seed: Dict[int, Dict[str, object]] = {}

    seeds_list = _resolve_seed_list(args)
    for seed in seeds_list:
        record, details = run_single(args, seed)
        run_records.append(record)
        details_by_seed[int(seed)] = details

    agg_records = aggregate(run_records)

    runs_csv = out_dir / f"{args.output_prefix}_runs_{args.mode}.csv"
    agg_csv = out_dir / f"{args.output_prefix}_aggregate_{args.mode}.csv"

    write_csv(runs_csv, [asdict(r) for r in run_records])
    write_csv(agg_csv, [asdict(r) for r in agg_records])

    plot_files: Dict[int, str] = {}
    diagnostic_csvs: Dict[int, str] = {}
    decision_audit_csvs: Dict[int, str] = {}
    checkpoint_leader_summary_csvs: Dict[int, str] = {}
    single_run = len(seeds_list) == 1
    for rec in run_records:
        if single_run:
            png = out_dir / f"{args.output_prefix}.png"
        else:
            png = out_dir / f"{args.output_prefix}_seed{rec.seed}_{args.mode}.png"
        det = details_by_seed[rec.seed]
        _plot_seed_trajectory(
            output_file=png,
            sample_interval=int(args.plot_sample_interval),
            top_series=np.asarray(det["top_series"], dtype=float),
            ex_leader_series=np.asarray(det["ex_leader_followers_series"], dtype=float),
            pu_share_series=np.asarray(det["pu_share_series"], dtype=float),
            welfare_series=np.asarray(det["welfare_series"], dtype=float),
            record=rec,
        )
        plot_files[rec.seed] = str(png)

        diag_rows = det.get("exit_diagnostics", [])
        if diag_rows:
            diag_csv = out_dir / f"{args.output_prefix}_seed{rec.seed}_{args.mode}_exit_diagnostics.csv"
            write_csv(diag_csv, diag_rows)
            diagnostic_csvs[rec.seed] = str(diag_csv)

        decision_audit_rows = det.get("decision_audit_rows", [])
        if decision_audit_rows:
            audit_csv = out_dir / f"{args.output_prefix}_seed{rec.seed}_{args.mode}_decision_audit.csv"
            write_csv(audit_csv, decision_audit_rows)
            decision_audit_csvs[rec.seed] = str(audit_csv)

        checkpoint_summary_rows = det.get("decision_audit_checkpoint_summary_rows", [])
        if checkpoint_summary_rows:
            summary_csv = out_dir / f"{args.output_prefix}_seed{rec.seed}_{args.mode}_checkpoint_leader_summary.csv"
            write_csv(summary_csv, checkpoint_summary_rows)
            checkpoint_leader_summary_csvs[rec.seed] = str(summary_csv)

    return {
        "args": args,
        "output_dir": str(out_dir),
        "run_records": run_records,
        "aggregate_records": agg_records,
        "details_by_seed": details_by_seed,
        "runs_csv": str(runs_csv),
        "aggregate_csv": str(agg_csv),
        "plot_files": plot_files,
        "diagnostic_csvs": diagnostic_csvs,
        "decision_audit_csvs": decision_audit_csvs,
        "checkpoint_leader_summary_csvs": checkpoint_leader_summary_csvs,
    }


def _print_summary(result: Dict[str, object]) -> None:
    run_records: List[RunRecord] = result["run_records"]  # type: ignore[assignment]
    agg_records: List[AggregateRecord] = result["aggregate_records"]  # type: ignore[assignment]

    print("\n" + "#" * 72)
    print("Experiment D: Perturbation/Recovery Summary")
    print("#" * 72)
    print(f"output dir: {result['output_dir']}")
    print(f"runs csv: {result['runs_csv']}")
    print(f"aggregate csv: {result['aggregate_csv']}")
    for seed, plot in result["plot_files"].items():
        print(f"plot (seed {seed}): {plot}")
    for seed, diag_csv in result.get("diagnostic_csvs", {}).items():
        print(f"exit diagnostics (seed {seed}): {diag_csv}")
    for seed, audit_csv in result.get("decision_audit_csvs", {}).items():
        print(f"decision audit (seed {seed}): {audit_csv}")
    for seed, summary_csv in result.get("checkpoint_leader_summary_csvs", {}).items():
        print(f"checkpoint leader summary (seed {seed}): {summary_csv}")

    print("\nPer-seed:")
    for r in run_records:
        print(
            f"seed={r.seed:>3} converged={int(r.converged)} t_conv={r.t_conv:>5} "
            f"leader_pre={r.leader_pre:>3} pre_followers={r.pre_followers:>3} "
            f"drop_fraction={r.drop_fraction:.3f} normless_dur={r.normless_duration:>5} "
            f"recovery_time={r.recovery_time:>5} stable={int(r.stable_recovery)} "
            f"final_leader={r.final_leader:>3} "
            f"leader_changed_final={int(r.final_leader_changed)} final_top={r.final_top_followers:>3}"
        )

    if agg_records:
        a = agg_records[0]
        print("\nAggregate:")
        print(
            f"conv_rate={a.conv_rate:.3f} drop_rate={a.drop_rate:.3f} "
            f"normless_rate={a.normless_rate:.3f} recovery_rate={a.recovery_rate:.3f} "
            f"stable_recovery_rate={a.stable_recovery_rate:.3f}"
        )
        print(
            f"mean_drop_fraction={a.mean_drop_fraction:.3f} ± {a.ci95_drop_fraction:.3f} (95% CI), "
            f"mean_normless_duration={a.mean_normless_duration:.1f} ± {a.ci95_normless_duration:.1f}"
        )


def main() -> None:
    args = parse_args()

    result = run_experiment(
        mode=args.mode,
        num_agents=args.num_agents,
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_steps_max=args.num_steps_max,
        gamma=args.gamma,
        kappa=args.kappa,
        B_R=args.B_R,
        B_F=args.B_F,
        c_threshold=args.c_threshold,
        delta=args.delta,
        seeds=args.seeds,
        seed_start=args.seed_start,
        selected_seeds=args.selected_seeds,
        role_update_s0=args.role_update_s0,
        role_update_T_seq=args.role_update_T_seq,
        role_update_base_interval=args.role_update_base_interval,
        fixed_role_update_interval=args.fixed_role_update_interval,
        role_update_epochs=args.role_update_epochs,
        async_role_update_prob=args.async_role_update_prob,
        perturb_strength=args.perturb_strength,
        perturb_duration=args.perturb_duration,
        perturb_policy_mode=args.perturb_policy_mode,
        collapse_followers_on_perturb=args.collapse_followers_on_perturb,
        reputation_shock_factor=args.reputation_shock_factor,
        post_window=args.post_window,
        conv_threshold=args.conv_threshold,
        conv_hold_steps=args.conv_hold_steps,
        recovery_threshold=args.recovery_threshold,
        recovery_hold_steps=args.recovery_hold_steps,
        stable_tail_window=args.stable_tail_window,
        dominant_threshold=args.dominant_threshold,
        drop_fraction_threshold=args.drop_fraction_threshold,
        tail_window=args.tail_window,
        output_dir=args.output_dir,
        run_label=args.run_label,
        auto_run_subdir=args.auto_run_subdir,
        plot_sample_interval=args.plot_sample_interval,
        tracking_mode=args.tracking_mode,
        initial_actor_rate=args.initial_actor_rate,
        initial_participant_rate=args.initial_participant_rate,
        reward_model=args.reward_model,
        reward_base_mu=args.reward_base_mu,
        reward_base_sigma=args.reward_base_sigma,
        reward_agent_sigma=args.reward_agent_sigma,
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
        reward_good_value=args.reward_good_value,
        reward_bad_value=args.reward_bad_value,
        reward_order_gap=args.reward_order_gap,
        numpy_fast_path=args.numpy_fast_path,
        output_prefix=args.output_prefix,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()
