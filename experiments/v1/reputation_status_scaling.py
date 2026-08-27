"""
Merged reputation x status scaling harness (gamma x kappa grid).

Built from status_scaling.py's per-run metrics (already correctly typed for
status/kappa, and the only one of the two original harnesses with a working
norm-optimality check), generalized from a single fixed --gamma to a swept
--gammas list so every cell of the grid gets the same treatment.

Goals:
1. Study how status incentives affect leader formation and stability, at
   varying reputation strength
2. Measure whether status leads to welfare improvements
3. Check whether emergent norms are welfare-optimal

Fixes applied relative to the two source files:
- status_scaling.py declared --c-threshold/--B-R/--B-F on the CLI but never
  wired them into SystemConfig (make_config hardcoded 0.1/0.3/0.2 regardless
  of what was passed). Fixed here -- these flags now do what they say.
- kappa is swept as a full grid against gamma via itertools.product, not a
  zip (which silently truncates to the shorter list).
- kappa enters the model through Ĵ^s, a SUM over followers (O(N)), while
  gamma enters through Ĵ^r (O(1)). Sweeping raw kappa on the same numeric
  scale as gamma is "status off" vs "status saturated" with nothing in
  between at realistic N. --kappa-scale-by-n divides by N before it reaches
  the engine; the raw value is what gets recorded.
- time_to_90pct_followers silently dropped runs that never reached the
  threshold from its mean, with no visibility into how many were dropped.
  reach_rate/n_reached now report that alongside the mean.
- leader_switches used lowest-agent-id tie-breaking, which manufactures
  "switches" out of near-ties -- and status makes near-ties MORE common
  (multiple agents clearing the follower gate at once). --leader-switch-margin
  requires a challenger to strictly exceed the incumbent before it counts.
"""

from __future__ import annotations

import argparse
import csv
import io
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.code_debugged import MultiAgentSystem, SystemConfig  # noqa: E402


# Output records
@dataclass
class RunRecord:
    mode: str
    gamma: float
    kappa: float
    seed: int
    leader_id: int
    final_top_followers: int
    time_to_90pct_followers: int
    leader_switches: int
    tail_welfare: float
    leader_role_final: str
    leader_is_status_final: int
    final_status_count: int
    tail_status_leader_share: float
    tail_status_agent_share: float

    final_norm: str
    best_norm: str
    final_norm_welfare_check: float
    best_norm_welfare: float
    welfare_gap_to_best: float
    is_final_norm_optimal: int
    consensus_step_first: int
    consensus_step_final: int
    num_consensus: int
    leader_actor_rates: list

    # --- convergence censoring bookkeeping ---
    follower_threshold: int
    reached_follower_threshold: int
    tail_top_follower_share: float

@dataclass
class AggregateRecord:
    mode: str
    gamma: float
    kappa: float
    n_runs: int
    reach_rate: float
    n_reached: int
    mean_final_top_followers: float
    std_final_top_followers: float
    ci95_final_top_followers: float
    mean_time_to_90pct_followers: float
    std_time_to_90pct_followers: float
    ci95_time_to_90pct_followers: float
    mean_leader_switches: float
    std_leader_switches: float
    ci95_leader_switches: float
    mean_tail_welfare: float
    std_tail_welfare: float
    ci95_tail_welfare: float
    mean_leader_is_status_final: float
    std_leader_is_status_final: float
    ci95_leader_is_status_final: float
    mean_final_status_count: float
    std_final_status_count: float
    ci95_final_status_count: float
    mean_tail_status_leader_share: float
    std_tail_status_leader_share: float
    ci95_tail_status_leader_share: float
    mean_tail_status_agent_share: float
    std_tail_status_agent_share: float
    ci95_tail_status_agent_share: float
    
    mean_final_norm_welfare_check: float
    std_final_norm_welfare_check: float
    ci95_final_norm_welfare_check: float
    mean_best_norm_welfare: float
    std_best_norm_welfare: float
    ci95_best_norm_welfare: float
    mean_welfare_gap_to_best: float
    std_welfare_gap_to_best: float
    ci95_welfare_gap_to_best: float
    mean_is_final_norm_optimal: float
    std_is_final_norm_optimal: float
    ci95_is_final_norm_optimal: float
    mean_tail_top_follower_share: float
    std_tail_top_follower_share: float
    ci95_tail_top_follower_share: float
    mean_consensus_step_first: float
    std_consensus_step_first: float
    ci95_consensus_step_first: float
    mean_consensus_step_final: float
    std_consensus_step_final: float
    ci95_consensus_step_final: float
    mean_num_consensus: float
    std_num_consensus: float
    ci95_num_consensus: float


Cell = Tuple[float, float]


def cell_tag(gamma: float, kappa: float) -> str:
    return f"g{_format_num(gamma)}_k{_format_num(kappa)}"


def cell_label(gamma: float, kappa: float) -> str:
    return f"gamma={gamma:g}, kappa={kappa:g}"


def _format_num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p").replace("-", "m")


@dataclass
class SeedComparisonRecord:
    mode: str
    gamma: float
    seed: int
    kappa: float
    leader_id: int
    leader_role_final: str
    final_top_followers: int
    tail_welfare: float
    final_norm: str
    final_norm_welfare_check: float
    best_norm: str
    best_norm_welfare: float
    welfare_gap_to_best: float
    is_final_norm_optimal: int


# Command-line configuration
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Status scaling kappa sweep harness.")
    parser.add_argument("--mode", choices=["static", "async"], required=True)
    parser.add_argument(
        "--gammas",
        type=str,
        default="5.0",
        help="Comma-separated gamma values. Swept as a FULL GRID against --kappas.",
    )
    parser.add_argument("--kappas", type=str, default="0,0.01,0.02,0.05,0.1")
    parser.add_argument(
        "--kappa-scale-by-n",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret --kappas as kappa-tilde and pass kappa = kappa_tilde / num_agents "
             "to the engine, since J^s is a SUM over followers (O(N)) while J^r is O(1). "
             "The raw kappa-tilde value is what gets recorded.",
    )
    parser.add_argument(
        "--convergence-threshold-frac",
        type=float,
        default=0.90,
        help="Follower fraction (of N-1) defining convergence to an opinion leader. "
             "Lower this if high-kappa cells cannot reach the bar because STATUS agents "
             "are ineligible to be followers.",
    )
    parser.add_argument(
        "--leader-switch-margin",
        type=int,
        default=1,
        help="Followers by which a challenger must STRICTLY EXCEED the incumbent before "
             "a leader switch is counted. 0 reproduces the original lowest-agent-id "
             "tie-breaking (which inflates leader_switches under near-ties).",
    )
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (inclusive).")
    parser.add_argument(
        "--selected-seeds",
        type=str,
        default="",
        help="Optional explicit comma-separated seed list (e.g. \"2,7,9\"). Overrides --seeds/--seed-start.",
    )
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument(
        "--actor-rate-driver-mode",
        choices=["standard", "status_if_followers_kappa0"],
        default="standard",
        help="Actor-rate driver mode for Eq. (13): paper-faithful standard or experimental status override at kappa=0.",
    )
    parser.add_argument(
        "--actor-rate-status-override-min-followers",
        type=int,
        default=10,
        help="Follower-count threshold for the experimental status-driven actor-rate override.",
    )
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--tail-window", type=int, default=500)
    parser.add_argument(
        "--role-update-s0",
        type=int,
        default=0,
        help="Paper notation s_0 for role-update epochs.",
    )
    parser.add_argument(
        "--role-update-T-seq",
        type=str,
        default="",
        help="Paper notation interval sequence T_n as comma-separated positive ints "
             "(e.g., \"2000,3000,6000\"); epochs are s_n = s_{n-1}+T_n. "
             "In async mode this is used as each agent's local interval progression.",
    )
    parser.add_argument("--role-update-base-interval", type=int, default=3000)
    parser.add_argument(
        "--fixed-role-update-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use constant role-update epochs T_n = const (Section 7.1.4) when enabled.",
    )
    parser.add_argument(
        "--plot-sample-interval",
        type=int,
        default=1,
        help="Downsample plotted time-series to every N steps (1 records every timestep).",
    )
    parser.add_argument(
        "--role-update-epochs",
        type=str,
        default="",
        help="Optional direct role-update epochs s_n as comma-separated positive ints "
             "(e.g., \"2000,3000,6000\"). Used when role_update_T_seq is empty.",
    )
    parser.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    parser.add_argument(
        "--force-all-active-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug override: force all agents to be active as both actors and participants every step.",
    )
    parser.add_argument("--initial-actor-rate", type=float, default=0.2)
    parser.add_argument("--initial-participant-rate", type=float, default=0.2)
    parser.add_argument(
        "--reward-model",
        choices=[
            "simple_preferred_action",
            "shared_base_gaussian",
            "shared_good_bad_heterogeneous",
            "consensus_welfare_gaussian",
        ],
        default="simple_preferred_action",
        help="Reward model for payoff generation.",
    )
    parser.add_argument("--reward-base-mu", type=float, default=0.5)
    parser.add_argument("--reward-base-sigma", type=float, default=0.08)
    parser.add_argument("--reward-agent-sigma", type=float, default=0.1)
    parser.add_argument("--reward-clip-min", type=float, default=0.01)
    parser.add_argument("--reward-clip-max", type=float, default=2.5)
    parser.add_argument(
        "--eq9-averaging-mode",
        choices=["participants_only", "all_agents"],
        default="participants_only",
        help="Eq. (9) observed-reputation averaging set.",
    )
    parser.add_argument(
        "--leader-update-mode",
        choices=["participants_only_post_eq9", "all_agents_post_eq9", "participants_only_pre_eq9"],
        default="participants_only_post_eq9",
        help="Section 6.4.4 timing/scope for updating L_i(t+1).",
    )
    parser.add_argument(
        "--numpy-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vectorized reputation updates in code_debugged.",
    )
    parser.add_argument(
        "--async-role-update-prob",
        type=float,
        default=None,
        help="Optional per-step Bernoulli probability for role updates in async mode. "
             "If omitted, async uses independent per-agent clocks driven by "
             "role_update_T_seq/role_update_epochs (or role_update_base_interval as fallback).",
    )
    parser.add_argument("--c-threshold", type=float, default=0.1)
    parser.add_argument("--B-R", dest="B_R", type=float, default=0.8)
    parser.add_argument("--B-F", dest="B_F", type=float, default=0.6)
    parser.add_argument(
        "--plot-leader-actor-rate",
        choices=["off", "per_cell", "grid", "overlay", "all"],
        default="off",
        help="Leader actor-interaction-rate vs step plots. 'per_cell': one file per "
             "(gamma,kappa), one line per seed. 'grid': single figure, subplot grid "
             "(rows=kappa, cols=gamma) of mean+-CI traces, shared axes for comparison. "
             "'overlay': single figure, one mean line per cell superimposed. 'all': all three.",
    )
    parser.add_argument(
        "--plot-leader-actor-rate-grid-show-seeds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In 'grid'/'all' layout, plot individual per-seed traces (thin, alpha) "
             "instead of mean+-CI. Gets noisy with many seeds/cells.",
    )
    return parser.parse_args()


# Small parsing helpers
def parse_kappas(kappa_text: str) -> List[float]:
    parts = [p.strip() for p in kappa_text.split(",") if p.strip()]
    return [float(x) for x in parts]


def parse_selected_seeds(seed_text: str) -> List[int]:
    if not seed_text.strip():
        return []
    parts = [p.strip() for p in seed_text.split(",") if p.strip()]
    seeds = [int(x) for x in parts]
    return sorted(set(seed for seed in seeds if seed >= 0))


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


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    selected = parse_selected_seeds(getattr(args, "selected_seeds", ""))
    if selected:
        return selected
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))


# System configuration for Experiment C
def effective_kappa(args: argparse.Namespace, kappa: float) -> float:
    """Map swept kappa-tilde onto the kappa actually handed to the engine."""
    if bool(getattr(args, "kappa_scale_by_n", False)):
        return float(kappa) / float(args.num_agents)
    return float(kappa)


def make_config(args: argparse.Namespace, gamma: float, kappa: float, mode: str) -> SystemConfig:
    role_interval = args.role_update_base_interval
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)
    eq9_averaging_mode = getattr(args, "eq9_averaging_mode", "participants_only")
    leader_update_mode = getattr(args, "leader_update_mode", "participants_only_post_eq9")
    actor_rate_driver_mode = getattr(args, "actor_rate_driver_mode", "standard")
    actor_rate_status_override_min_followers = int(
        getattr(args, "actor_rate_status_override_min_followers", 10)
    )
    if mode == "async":
        role_interval = args.num_steps + 1_000_000
        role_s0 = 0
        role_t_seq = []
        role_epochs = []

    return SystemConfig(
        num_agents=args.num_agents,
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_time_steps=args.num_steps,
        M=1.0,
        u_0=0.1,
        actor_rate_driver_mode=actor_rate_driver_mode,
        actor_rate_status_override_min_followers=actor_rate_status_override_min_followers,
        gamma=float(gamma),
        # FIX: the CLI declares --c-threshold/--B-R/--B-F but the original
        # make_config ignored args and hardcoded these three regardless of
        # what was passed. That flag was doing nothing.
        kappa=effective_kappa(args, kappa),
        c_threshold=float(getattr(args, "c_threshold", 0.1)),
        B_R=float(getattr(args, "B_R", 0.3)),
        B_F=float(getattr(args, "B_F", 0.2)),
        delta=args.delta,
        eq9_averaging_mode=eq9_averaging_mode,
        leader_update_mode=leader_update_mode,
        alpha_pu_base=0.05,
        beta_status_base=0.05,
        eta_v_base=0.1,
        eta_s_base=0.1,
        eta_J_base=0.05,
        role_update_s0=role_s0,
        role_update_T_sequence=role_t_seq,
        role_update_base_interval=role_interval,
        fixed_role_update_interval=args.fixed_role_update_interval,
        role_update_epochs=role_epochs,
        gossip_rate=0.5,
        gossip_alpha=0.5,
        tracking_mode=args.tracking_mode,
        use_numpy_fast_path=args.numpy_fast_path,
        force_all_active_debug=getattr(args, "force_all_active_debug", False),
        initial_actor_interaction_rate=args.initial_actor_rate,
        initial_participant_interaction_rate=args.initial_participant_rate,
        reward_model=args.reward_model,
        reward_base_mu=args.reward_base_mu,
        reward_base_sigma=args.reward_base_sigma,
        reward_agent_sigma=args.reward_agent_sigma,
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
    )


# Result extraction helpers 
def _finalize_results(system: MultiAgentSystem) -> Dict:
    final_roles = [a.state.role for a in system.agents]
    final_followers = [len(a.state.followers) for a in system.agents]
    opinion_leader = int(np.argmax(final_followers)) if max(final_followers) > 0 else -1

    results = dict(system.results)
    results["final_roles"] = final_roles
    results["final_followers"] = final_followers
    results["opinion_leader"] = opinion_leader
    return results


def _leader_series_from_follower_counts(follower_counts: np.ndarray) -> np.ndarray:
    leaders = np.full(shape=(follower_counts.shape[0],), fill_value=-1, dtype=int)
    for t in range(follower_counts.shape[0]):
        row = follower_counts[t]
        m = int(np.max(row))
        if m > 0:
            candidates = np.where(row == m)[0]
            leaders[t] = int(candidates[0])
    return leaders


def _leader_series_hysteretic(follower_counts: np.ndarray, margin: int) -> np.ndarray:
    """
    Leader identity with an incumbency margin. The plain series breaks ties by
    lowest agent id, so two agents trading a tie register as a switch every
    step -- and kappa makes near-ties MORE common (more agents clear the c*N
    status gate at comparable follower counts). Retain the incumbent unless a
    challenger strictly exceeds it by `margin` followers. margin=0 reproduces
    the original behaviour.
    """
    if margin <= 0:
        return _leader_series_from_follower_counts(follower_counts)
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


def _tail_top_follower_share(follower_counts: np.ndarray, tail_window: int, denom: int) -> float:
    if follower_counts.size == 0 or tail_window <= 0 or denom <= 0:
        return 0.0
    tail_window = min(tail_window, follower_counts.shape[0])
    tail = follower_counts[-tail_window:]
    return float(np.mean(tail.max(axis=1)) / float(denom))


def _leader_switches(leader_series: np.ndarray) -> int:
    non_null = [x for x in leader_series.tolist() if x >= 0]
    if len(non_null) <= 1:
        return 0
    return sum(1 for a, b in zip(non_null[:-1], non_null[1:]) if a != b)


def _time_to_threshold(series: np.ndarray, threshold: int) -> int:
    idx = np.where(series >= threshold)[0]
    return int(idx[0] + 1) if idx.size > 0 else -1


def _role_to_label(role) -> str:
    if hasattr(role, "value"):
        return str(role.value).lower()
    if hasattr(role, "name"):
        return str(role.name).lower()
    return str(role).lower()


def _tail_status_leader_share(role_history: np.ndarray, leader_series: np.ndarray, tail_window: int) -> float:
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
            if _role_to_label(role_history[t, leader]) == "status":
                hits += 1
    return float(hits / valid) if valid > 0 else 0.0


def _tail_status_agent_share(role_history: np.ndarray, tail_window: int) -> float:
    if role_history.size == 0 or tail_window <= 0:
        return 0.0
    tail_window = min(tail_window, role_history.shape[0])
    tail = role_history[-tail_window:]
    tail_labels = np.vectorize(_role_to_label)(tail)
    return float(np.mean(tail_labels == "status"))


def _deterministic_norm_welfare(system: MultiAgentSystem, norm_actions: Sequence[int], leader_id: int) -> float:
    """
    Evaluate followers-only paper welfare for a deterministic norm:
    norm_actions[s] is the chosen action in state s.
    """
    p_s = np.ones(system.config.num_states, dtype=float) / float(system.config.num_states)
    total = 0.0

    for i, agent in enumerate(system.agents):
        if i == int(leader_id):
            continue

        theta_participant = 1.0 - np.exp(-float(agent.state.participant_interaction_rate))

        U_i = 0.0
        for s in range(system.config.num_states):
            action = int(norm_actions[s])
            u_i_sx = system.compute_observer_utility(i, s, action)
            U_i += float(p_s[s]) * float(u_i_sx)

        total += theta_participant * U_i

    return float(total)


def _leader_greedy_norm(system: MultiAgentSystem, leader_id: int) -> Tuple[int, ...]:
    """
    Convert the final leader's role-consistent policy into a deterministic norm
    by taking argmax action in each state.
    """
    leader = system.agents[int(leader_id)]
    actions = []
    for s in range(system.config.num_states):
        pi_s = leader.get_current_policy(s)
        actions.append(int(np.argmax(pi_s)))
    return tuple(actions)


def _bruteforce_best_norm(system: MultiAgentSystem, leader_id: int) -> Tuple[Tuple[int, ...], float]:
    """
    Exhaustively search all deterministic norms.
    Only feasible for small num_actions ** num_states.
    """
    num_states = int(system.config.num_states)
    num_actions = int(system.config.num_actions)

    total_norms = num_actions ** num_states
    if total_norms > 50000:
        raise ValueError(
            f"Bruteforce norm search too large: {num_actions}^{num_states} = {total_norms}. "
            "Use only for small state/action spaces."
        )

    best_norm = None
    best_welfare = -np.inf

    for norm_actions in itertools.product(range(num_actions), repeat=num_states):
        welfare = _deterministic_norm_welfare(system, norm_actions, leader_id)
        if welfare > best_welfare:
            best_welfare = welfare
            best_norm = tuple(int(x) for x in norm_actions)

    return best_norm, float(best_welfare)


def _final_norm_optimality_check(system: MultiAgentSystem, leader_id: int) -> Dict[str, object]:
    """
    Compare final leader greedy norm to brute-force best deterministic norm.
    """
    if leader_id < 0:
        return {
            "final_norm": (),
            "best_norm": (),
            "final_norm_welfare_check": float("nan"),
            "best_norm_welfare": float("nan"),
            "welfare_gap_to_best": float("nan"),
            "is_final_norm_optimal": 0,
        }
    
    final_norm = _leader_greedy_norm(system, leader_id)
    final_welfare = _deterministic_norm_welfare(system, final_norm, leader_id)
    best_norm, best_welfare = _bruteforce_best_norm(system, leader_id)

    gap = float(best_welfare - final_welfare)
    is_optimal = int(tuple(final_norm) == tuple(best_norm) or np.isclose(gap, 0.0, atol=1e-10, rtol=0.0))

    return {
        "final_norm": final_norm,
        "best_norm": best_norm,
        "final_norm_welfare_check": float(final_welfare),
        "best_norm_welfare": float(best_welfare),
        "welfare_gap_to_best": float(gap),
        "is_final_norm_optimal": int(is_optimal),
    }


def _norm_to_str(norm: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in norm)


# Single simulation run
def run_single(args: argparse.Namespace, mode: str, gamma: float, kappa: float, seed: int) -> RunRecord:
    np.random.seed(seed)
    config = make_config(args, gamma=gamma, kappa=kappa, mode=mode)
    system = MultiAgentSystem(config)

    consensus_step_first = 0
    consensus_step_final = 0
    cur_consensus = False
    num_consensus = 0
    actor_rates = {i: [] for i,a in enumerate(system.agents)}

    if mode == "async":
        with redirect_stdout(io.StringIO()):
            if args.async_role_update_prob is None:
                interval_seq, async_s0, _ = _build_async_interval_sequence(args)
                first_interval = int(interval_seq[0])
                role_timers = np.random.randint(1, first_interval + 1, size=args.num_agents, dtype=int)
                if async_s0 > 0:
                    role_timers = role_timers + async_s0
                interval_indices = np.zeros(args.num_agents, dtype=int)
            else:
                async_update_prob = float(args.async_role_update_prob)

            for _ in range(args.num_steps):
                system.step()
                if args.async_role_update_prob is None:
                    role_timers -= 1
                    update_ids = np.where(role_timers <= 0)[0]
                    if update_ids.size > 0:
                        update_list = update_ids.tolist()
                        system._update_roles_sequential(update_list)
                        system.refresh_last_tracked_state()
                        system.results.setdefault("role_update_times", []).append(int(system.time_step))
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
                    update_mask = np.random.random(args.num_agents) < async_update_prob
                    update_ids = np.where(update_mask)[0]
                    if update_ids.size > 0:
                        system._update_roles_sequential(update_ids.tolist())
                        system.refresh_last_tracked_state()
                        system.results.setdefault("role_update_times", []).append(int(system.time_step))
                        system.role_update_epoch += 1
                updated_consensus = np.max([len(a.state.followers) for a in system.agents]) >= 0.50 * len(system.agents)
                if consensus_step_first == 0 and updated_consensus:
                    consensus_step_first = _ + 1
                elif not cur_consensus and updated_consensus:
                    consensus_step_final = _ + 1
                    num_consensus += 1
                    cur_consensus = True
                elif not updated_consensus:
                    cur_consensus = False
                _ = [actor_rates[i].append(a.state.actor_interaction_rate) for i,a in enumerate(system.agents)]
            
            results = _finalize_results(system)
            results["consensus_step_first"] = consensus_step_first
            results["consensus_step_final"] = consensus_step_final
            results["num_consensus"] = num_consensus

    else:
        with redirect_stdout(io.StringIO()):
            for _ in range(args.num_steps):
                system.step()
                updated_consensus = np.max([len(a.state.followers) for a in system.agents]) >= 0.50 * len(system.agents)
                if consensus_step_first == 0 and updated_consensus:
                    consensus_step_first = _ + 1
                elif not cur_consensus and updated_consensus:
                    consensus_step_final = _ + 1
                    num_consensus += 1
                    cur_consensus = True
                elif not updated_consensus:
                    cur_consensus = False
                _ = [actor_rates[i].append(a.state.actor_interaction_rate) for i,a in enumerate(system.agents)]
            results = _finalize_results(system)
            results["consensus_step_first"] = consensus_step_first
            results["consensus_step_final"] = consensus_step_final
            results["num_consensus"] = num_consensus


    follower_counts = np.asarray(results["follower_counts"], dtype=float)
    role_history = np.asarray(results.get("role_label_history", []), dtype=object)
    social_welfare = np.asarray(results.get("paper_welfare_followers_only", []), dtype=float)

    top_follower_series = follower_counts.max(axis=1)
    # Plain series (lowest-id tie-break) kept for plotting continuity; the
    # hysteretic series is what leader_switches is computed from.
    leader_series = _leader_series_from_follower_counts(follower_counts)
    leader_series_stable = _leader_series_hysteretic(
        follower_counts, margin=int(getattr(args, "leader_switch_margin", 1))
    )

    threshold_frac = float(getattr(args, "convergence_threshold_frac", 0.90))
    follower_threshold = int(np.ceil(threshold_frac * (args.num_agents - 1)))
    time_to_90 = _time_to_threshold(top_follower_series, follower_threshold)
    leader_switches = _leader_switches(leader_series_stable)

    tail_window = min(int(args.tail_window), len(social_welfare)) if len(social_welfare) > 0 else 0
    tail_welfare = float(np.mean(social_welfare[-tail_window:])) if tail_window > 0 else float("nan")

    leader_id = int(results["opinion_leader"])
    final_roles = [_role_to_label(r) for r in results["final_roles"]]
    leader_role_final = final_roles[leader_id] if leader_id >= 0 else "none"
    leader_is_status_final = int(leader_role_final == "status")
    final_status_count = sum(1 for r in final_roles if r == "status")
    tail_status_leader_share = _tail_status_leader_share(role_history, leader_series_stable, tail_window)
    tail_status_agent_share = _tail_status_agent_share(role_history, tail_window)
    tail_top_follower_share = _tail_top_follower_share(
        follower_counts, tail_window, denom=int(args.num_agents) - 1
    )

    try:
        optimality = _final_norm_optimality_check(system, leader_id)
    except ValueError:
        optimality = {
            "final_norm": (),
            "best_norm": (),
            "final_norm_welfare_check": float("nan"),
            "best_norm_welfare": float("nan"),
            "welfare_gap_to_best": float("nan"),
            "is_final_norm_optimal": -1,
        }

    return RunRecord(
        mode=mode,
        gamma=float(gamma),
        kappa=float(kappa),
        seed=int(seed),
        leader_id=leader_id,
        final_top_followers=int(max(results["final_followers"])),
        time_to_90pct_followers=int(time_to_90),
        leader_switches=int(leader_switches),
        tail_welfare=float(tail_welfare),
        follower_threshold=int(follower_threshold),
        reached_follower_threshold=int(time_to_90 >= 0),
        tail_top_follower_share=float(tail_top_follower_share),
        leader_role_final=str(leader_role_final),
        leader_is_status_final=int(leader_is_status_final),
        final_status_count=int(final_status_count),
        tail_status_leader_share=float(tail_status_leader_share),
        tail_status_agent_share=float(tail_status_agent_share),

        final_norm=_norm_to_str(optimality["final_norm"]) if optimality["final_norm"] else "",
        best_norm=_norm_to_str(optimality["best_norm"]) if optimality["best_norm"] else "",
        final_norm_welfare_check=float(optimality["final_norm_welfare_check"]),
        best_norm_welfare=float(optimality["best_norm_welfare"]),
        welfare_gap_to_best=float(optimality["welfare_gap_to_best"]),
        is_final_norm_optimal=int(optimality["is_final_norm_optimal"]),
        consensus_step_first=results["consensus_step_first"],
        consensus_step_final=results["consensus_step_final"],
        num_consensus=results["num_consensus"],
        leader_actor_rates=actor_rates[leader_id] if leader_id > -1 else [0] * len(actor_rates[0])
    )


# Aggregation and saving
def _mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size >= 2 else 0.0
    return mean, std, ci95


def aggregate(records: Sequence[RunRecord]) -> List[AggregateRecord]:
    grouped: Dict[Tuple[str, float, float], List[RunRecord]] = {}
    for r in records:
        grouped.setdefault((r.mode, r.gamma, r.kappa), []).append(r)

    rows: List[AggregateRecord] = []
    for (mode, gamma, kappa), recs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        m1, s1, c1 = _mean_std_ci([r.final_top_followers for r in recs])
        reached_vals = [r.time_to_90pct_followers for r in recs if r.time_to_90pct_followers >= 0]
        n_reached = len(reached_vals)
        # mean_time_to_90pct_followers below is CONDITIONAL ON REACHING, and
        # whether a run reaches is itself a function of gamma/kappa. Read
        # reach_rate alongside it -- never read the mean alone.
        m2, s2, c2 = _mean_std_ci(reached_vals if reached_vals else [-1.0])
        m3, s3, c3 = _mean_std_ci([r.leader_switches for r in recs])
        m_tfs, s_tfs, c_tfs = _mean_std_ci([r.tail_top_follower_share for r in recs])
        m4, s4, c4 = _mean_std_ci([r.tail_welfare for r in recs])
        m5, s5, c5 = _mean_std_ci([r.leader_is_status_final for r in recs])
        m6, s6, c6 = _mean_std_ci([r.final_status_count for r in recs])
        m7, s7, c7 = _mean_std_ci([r.tail_status_leader_share for r in recs])
        m8, s8, c8 = _mean_std_ci([r.tail_status_agent_share for r in recs])

        
        m9, s9, c9 = _mean_std_ci([r.final_norm_welfare_check for r in recs if np.isfinite(r.final_norm_welfare_check)])
        m10, s10, c10 = _mean_std_ci([r.best_norm_welfare for r in recs if np.isfinite(r.best_norm_welfare)])
        m11, s11, c11 = _mean_std_ci([r.welfare_gap_to_best for r in recs if np.isfinite(r.welfare_gap_to_best)])
        m12, s12, c12 = _mean_std_ci([r.is_final_norm_optimal for r in recs if r.is_final_norm_optimal >= 0])
        m13, s13, c13 = _mean_std_ci([r.consensus_step_first for r in recs if r.consensus_step_first > 0])
        m14, s14, c14 = _mean_std_ci([r.consensus_step_final for r in recs if r.consensus_step_final > 0])
        m15, s15, c15 = _mean_std_ci([r.num_consensus for r in recs if r.num_consensus > 0])
        rows.append(
            AggregateRecord(
                mode=mode,
                gamma=float(gamma),
                kappa=float(kappa),
                n_runs=len(recs),
                reach_rate=float(n_reached) / float(len(recs)) if recs else 0.0,
                n_reached=int(n_reached),
                mean_final_top_followers=m1,
                std_final_top_followers=s1,
                ci95_final_top_followers=c1,
                mean_time_to_90pct_followers=m2,
                std_time_to_90pct_followers=s2,
                ci95_time_to_90pct_followers=c2,
                mean_leader_switches=m3,
                std_leader_switches=s3,
                ci95_leader_switches=c3,
                mean_tail_welfare=m4,
                std_tail_welfare=s4,
                ci95_tail_welfare=c4,
                mean_leader_is_status_final=m5,
                std_leader_is_status_final=s5,
                ci95_leader_is_status_final=c5,
                mean_final_status_count=m6,
                std_final_status_count=s6,
                ci95_final_status_count=c6,
                mean_tail_status_leader_share=m7,
                std_tail_status_leader_share=s7,
                ci95_tail_status_leader_share=c7,
                mean_tail_status_agent_share=m8,
                std_tail_status_agent_share=s8,
                ci95_tail_status_agent_share=c8,
                
                mean_final_norm_welfare_check=m9,
                std_final_norm_welfare_check=s9,
                ci95_final_norm_welfare_check=c9,
                mean_best_norm_welfare=m10,
                std_best_norm_welfare=s10,
                ci95_best_norm_welfare=c10,
                mean_welfare_gap_to_best=m11,
                std_welfare_gap_to_best=s11,
                ci95_welfare_gap_to_best=c11,
                mean_is_final_norm_optimal=m12,
                std_is_final_norm_optimal=s12,
                ci95_is_final_norm_optimal=c12,
                mean_tail_top_follower_share=m_tfs,
                std_tail_top_follower_share=s_tfs,
                ci95_tail_top_follower_share=c_tfs,
                mean_consensus_step_first=m13,
                std_consensus_step_first=s13,
                ci95_consensus_step_first=c13,
                mean_consensus_step_final=m14,
                std_consensus_step_final=s14,
                ci95_consensus_step_final=c14,
                mean_num_consensus=m15,
                std_num_consensus=s15,
                ci95_num_consensus=c15
            )
        )
    return rows


def build_seed_comparison(records: Sequence[RunRecord]) -> List[SeedComparisonRecord]:
    rows: List[SeedComparisonRecord] = []
    for r in sorted(records, key=lambda x: (x.mode, x.gamma, x.seed, x.kappa)):
        rows.append(
            SeedComparisonRecord(
                mode=r.mode,
                gamma=r.gamma,
                seed=r.seed,
                kappa=r.kappa,
                leader_id=r.leader_id,
                leader_role_final=r.leader_role_final,
                final_top_followers=r.final_top_followers,
                tail_welfare=r.tail_welfare,
                final_norm=r.final_norm,
                final_norm_welfare_check=r.final_norm_welfare_check,
                best_norm=r.best_norm,
                best_norm_welfare=r.best_norm_welfare,
                welfare_gap_to_best=r.welfare_gap_to_best,
                is_final_norm_optimal=r.is_final_norm_optimal,
            )
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Plotting helpers
def plot_metric(aggregate_rows: Sequence[AggregateRecord], field: str, ylabel: str, output_file: Path) -> None:
    """One line per gamma, means vs kappa -- the direct generalization of the
    original single-gamma plot to a gamma x kappa grid."""
    gammas = sorted({float(r.gamma) for r in aggregate_rows})
    plt.figure(figsize=(6.4, 4.2))
    for gamma in gammas:
        rows = sorted((r for r in aggregate_rows if float(r.gamma) == gamma), key=lambda r: r.kappa)
        if not rows:
            continue
        kappas = np.array([r.kappa for r in rows], dtype=float)
        means = np.array([getattr(r, field) for r in rows], dtype=float)
        plt.plot(kappas, means, "-o", linewidth=1.8, label=f"gamma={gamma:g}")
    plt.xlabel("kappa")
    plt.ylabel(ylabel)
    if len(gammas) > 1:
        plt.legend(fontsize=9)
    plt.grid(alpha=0.25)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def plot_kappa_heatmap(aggregate_rows: Sequence[AggregateRecord], field: str, title: str, output_file: Path) -> None:
    """gamma x kappa heatmap of one aggregate field."""
    gammas = sorted({float(r.gamma) for r in aggregate_rows})
    kappas = sorted({float(r.kappa) for r in aggregate_rows})
    if len(gammas) < 1 or len(kappas) < 1:
        return
    lookup = {(float(r.gamma), float(r.kappa)): getattr(r, field) for r in aggregate_rows}
    grid = np.full((len(kappas), len(gammas)), np.nan, dtype=float)
    for j, g in enumerate(gammas):
        for i, k in enumerate(kappas):
            val = lookup.get((g, k))
            if val is not None:
                grid[i, j] = float(val)
    fig, ax = plt.subplots(figsize=(1.1 * len(gammas) + 3.0, 0.9 * len(kappas) + 2.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(gammas)), [f"{g:g}" for g in gammas])
    ax.set_yticks(range(len(kappas)), [f"{k:g}" for k in kappas])
    ax.set_xlabel("gamma")
    ax.set_ylabel("kappa")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(kappas)):
        for j in range(len(gammas)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3g}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def _errorbar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    yerr: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    ylim=None,
):
    plt.figure(figsize=(7, 4.6))
    plt.errorbar(
        df[x],
        df[y],
        yerr=df[yerr],
        marker="o",
        capsize=4,
        linewidth=1.8,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing column among {candidates}")


def _errorbar_plot_multi_gamma(
    agg_df: pd.DataFrame,
    y: str,
    yerr: str,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim=None,
) -> None:
    """One errorbar line per gamma, x=kappa. Generalizes the original
    single-gamma _errorbar_plot now that the grid has more than one gamma."""
    plt.figure(figsize=(7, 4.6))
    for gamma, sub_df in agg_df.groupby("gamma"):
        sub_df = sub_df.sort_values("kappa")
        plt.errorbar(
            sub_df["kappa"], sub_df[y], yerr=sub_df[yerr],
            marker="o", capsize=4, linewidth=1.8, label=f"gamma={gamma:g}",
        )
    plt.title(title)
    plt.xlabel("$\\kappa$")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    if agg_df["gamma"].nunique() > 1:
        plt.legend(fontsize=9)
    plt.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_expC_report_figures(
    run_records: Sequence[RunRecord],
    aggregate_rows: Sequence[AggregateRecord],
    output_dir: Path,
) -> None:
    """
    Generate final report figures for Experiment C, one line per gamma.

    Outputs:
    - expC_leader_is_status_vs_kappa.png
    - expC_tail_welfare_vs_kappa.png
    - expC_welfare_gap_vs_kappa.png
    - expC_optimal_norm_probability_vs_kappa.png
    """

    fig_dir = output_dir.parent / "final_report_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    agg_df = pd.DataFrame([asdict(r) for r in aggregate_rows])
    if agg_df.empty:
        print("[!] No aggregate rows -- skipping report figures.")
        return

    _errorbar_plot_multi_gamma(
        agg_df, y="mean_leader_is_status_final", yerr="ci95_leader_is_status_final",
        title="Final leader is STATUS vs $\\kappa$",
        ylabel="Probability final leader is STATUS",
        out_path=fig_dir / "expC_leader_is_status_vs_kappa.png",
        ylim=(-0.05, 1.05),
    )
    _errorbar_plot_multi_gamma(
        agg_df, y="mean_tail_welfare", yerr="ci95_tail_welfare",
        title="Tail welfare vs $\\kappa$",
        ylabel="Mean tail welfare",
        out_path=fig_dir / "expC_tail_welfare_vs_kappa.png",
    )
    _errorbar_plot_multi_gamma(
        agg_df, y="mean_welfare_gap_to_best", yerr="ci95_welfare_gap_to_best",
        title="Welfare gap to best norm vs $\\kappa$",
        ylabel="Mean welfare gap",
        out_path=fig_dir / "expC_welfare_gap_vs_kappa.png",
    )
    _errorbar_plot_multi_gamma(
        agg_df, y="mean_is_final_norm_optimal", yerr="ci95_is_final_norm_optimal",
        title="Probability final norm is optimal vs $\\kappa$",
        ylabel="Probability optimal",
        out_path=fig_dir / "expC_optimal_norm_probability_vs_kappa.png",
        ylim=(-0.05, 1.05),
    )

    if agg_df["gamma"].nunique() > 1 and agg_df["kappa"].nunique() > 1:
        plot_kappa_heatmap(
            aggregate_rows, field="reach_rate",
            title="Fraction of seeds reaching the follower threshold",
            output_file=fig_dir / "expC_heatmap_reach_rate.png",
        )
        plot_kappa_heatmap(
            aggregate_rows, field="mean_leader_is_status_final",
            title="P(final leader is STATUS)",
            output_file=fig_dir / "expC_heatmap_leader_is_status.png",
        )

    print(f"[✓] Report figures saved to {fig_dir}")

def _leader_actor_rate_by_cell(records: Sequence[RunRecord]) -> Dict[Cell, List[RunRecord]]:
    grouped: Dict[Cell, List[RunRecord]] = {}
    for r in records:
        grouped.setdefault((r.gamma, r.kappa), []).append(r)
    return grouped


def _stacked_leader_actor_rate(recs: Sequence[RunRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate all seeds in a cell to the shortest trace and stack -> (n_seeds, min_len)."""
    traces = [np.asarray(r.leader_actor_rates, dtype=float) for r in recs]
    traces = [t for t in traces if t.size > 0]
    if not traces:
        return np.empty((0, 0)), np.empty((0,))
    min_len = min(t.size for t in traces)
    stacked = np.vstack([t[:min_len] for t in traces])
    steps = np.arange(1, min_len + 1)
    return stacked, steps


def plot_leader_actor_rate_per_cell(
    records: Sequence[RunRecord], output_dir: Path, mode: str, sample_interval: int = 1,
) -> None:
    """One file per (gamma, kappa) cell, one line per seed -- for inspecting
    individual runs within a cell."""
    fig_dir = output_dir / "leader_actor_rate"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sample_interval = max(1, int(sample_interval))

    for (gamma, kappa), recs in sorted(_leader_actor_rate_by_cell(records).items()):
        plt.figure(figsize=(7, 4.6))
        any_plotted = False
        for r in sorted(recs, key=lambda x: x.seed):
            rates = np.asarray(r.leader_actor_rates, dtype=float)
            if rates.size == 0:
                continue
            steps = np.arange(1, rates.size + 1)[::sample_interval]
            plt.plot(steps, rates[::sample_interval], linewidth=1.0, alpha=0.7, label=f"seed={r.seed}")
            any_plotted = True
        if not any_plotted:
            plt.close()
            continue
        plt.title(f"Leader actor interaction rate — {cell_label(gamma, kappa)} ({mode})")
        plt.xlabel("step")
        plt.ylabel("leader actor interaction rate")
        if len(recs) <= 12:
            plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / f"leader_actor_rate_{cell_tag(gamma, kappa)}_{mode}.png", dpi=160)
        plt.close()

    print(f"[✓] Per-cell leader actor-rate plots saved to {fig_dir}")


def plot_leader_actor_rate_grid(
    records: Sequence[RunRecord], output_dir: Path, mode: str,
    sample_interval: int = 1, show_seeds: bool = False,
) -> None:
    """Single figure, subplot grid (cols=gamma, rows=kappa) sharing x/y axes so
    traces are directly comparable across cells at a glance."""
    grouped = _leader_actor_rate_by_cell(records)
    if not grouped:
        return
    gammas = sorted({g for g, k in grouped})
    kappas = sorted({k for g, k in grouped})
    sample_interval = max(1, int(sample_interval))

    fig, axes = plt.subplots(
        len(kappas), len(gammas),
        figsize=(3.2 * len(gammas), 2.4 * len(kappas)),
        sharex=True, sharey=True, squeeze=False,
    )

    for i, kappa in enumerate(kappas):
        for j, gamma in enumerate(gammas):
            ax = axes[i][j]
            recs = grouped.get((gamma, kappa), [])
            if not recs:
                ax.axis("off")
                continue
            if show_seeds:
                for r in sorted(recs, key=lambda x: x.seed):
                    rates = np.asarray(r.leader_actor_rates, dtype=float)
                    if rates.size == 0:
                        continue
                    steps = np.arange(1, rates.size + 1)[::sample_interval]
                    ax.plot(steps, rates[::sample_interval], linewidth=0.8, alpha=0.5)
            else:
                stacked, steps_full = _stacked_leader_actor_rate(recs)
                if stacked.size > 0:
                    steps = steps_full[::sample_interval]
                    mean = stacked.mean(axis=0)[::sample_interval]
                    if stacked.shape[0] >= 2:
                        std = stacked.std(axis=0, ddof=1)[::sample_interval]
                        ci95 = 1.96 * std / np.sqrt(stacked.shape[0])
                        ax.fill_between(steps, mean - ci95, mean + ci95, alpha=0.25)
                    ax.plot(steps, mean, linewidth=1.5)
            if i == 0:
                ax.set_title(f"gamma={gamma:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"kappa={kappa:g}", fontsize=9)
            ax.grid(alpha=0.2)

    fig.suptitle(f"Leader actor interaction rate vs step ({mode})", fontsize=12, fontweight="bold")
    fig.supxlabel("step")
    fig.supylabel("leader actor interaction rate")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])

    fig_dir = output_dir / "leader_actor_rate"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"leader_actor_rate_grid_{mode}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[✓] Leader actor-rate grid saved to {out_path}")


def plot_leader_actor_rate_overlay(
    records: Sequence[RunRecord], output_dir: Path, mode: str, sample_interval: int = 1,
) -> None:
    """Single figure, one mean trace per (gamma, kappa) cell superimposed --
    for reading off cross-cell differences directly."""
    grouped = _leader_actor_rate_by_cell(records)
    if not grouped:
        return
    sample_interval = max(1, int(sample_interval))
    cells = sorted(grouped.keys())

    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for idx, (gamma, kappa) in enumerate(cells):
        stacked, steps_full = _stacked_leader_actor_rate(grouped[(gamma, kappa)])
        if stacked.size == 0:
            continue
        steps = steps_full[::sample_interval]
        mean = stacked.mean(axis=0)[::sample_interval]
        color = cmap(idx / max(1, len(cells) - 1))
        plt.plot(steps, mean, linewidth=1.6, color=color, label=cell_label(gamma, kappa))

    plt.title(f"Leader actor interaction rate vs step, all cells ({mode})")
    plt.xlabel("step")
    plt.ylabel("mean leader actor interaction rate")
    plt.legend(fontsize=7, ncol=2, loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig_dir = output_dir / "leader_actor_rate"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"leader_actor_rate_overlay_{mode}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[✓] Leader actor-rate overlay saved to {out_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gammas = parse_kappas(args.gammas)  # same "comma floats" parser works for either
    kappas = parse_kappas(args.kappas)
    seeds = resolve_seeds(args)
    cells: List[Cell] = [(float(g), float(k)) for g, k in itertools.product(gammas, kappas)]

    print("#" * 72)
    print("Reputation x Status Scaling Grid Run")
    print(f"mode={args.mode}")
    print(f"gammas={gammas}")
    print(f"kappas={kappas} (scale_by_n={bool(args.kappa_scale_by_n)})")
    if args.kappa_scale_by_n:
        print(f"  effective kappas passed to engine: {[k / args.num_agents for k in kappas]}")
    print(f"grid cells={len(gammas)}x{len(kappas)}={len(cells)}, seeds={len(seeds)}")
    print(f"c_threshold={args.c_threshold}, B_R={args.B_R}, B_F={args.B_F}")
    print(
        f"convergence_threshold_frac={args.convergence_threshold_frac}, "
        f"leader_switch_margin={args.leader_switch_margin}"
    )
    print("#" * 72)

    run_records: List[RunRecord] = []
    job, total_jobs = 0, len(cells) * len(seeds)
    for gamma, kappa in cells:
        for seed in seeds:
            job += 1
            record = run_single(args, mode=args.mode, gamma=gamma, kappa=kappa, seed=seed)
            run_records.append(record)

            print(
                f"[{job:03d}/{total_jobs:03d}] mode={record.mode} gamma={record.gamma:g} kappa={record.kappa:g} seed={record.seed} "
                f"leader={record.leader_id} top_followers={record.final_top_followers} "
                f"leader_role={record.leader_role_final} n_status={record.final_status_count} "
                f"tail_welfare={record.tail_welfare:.4f} "
                f"optimal={record.is_final_norm_optimal} gap={record.welfare_gap_to_best:.6f}"
                f"first_consensus= {record.consensus_step_first} "
                f"last_consensus= {record.consensus_step_final} "
                f"num_consensus= {record.num_consensus}"
            )

    aggregate_rows = aggregate(run_records)
    seed_comparison_rows = build_seed_comparison(run_records)

    run_csv = output_dir / f"reputation_status_scaling_runs_{args.mode}.csv"
    agg_csv = output_dir / f"reputation_status_scaling_aggregate_{args.mode}.csv"
    seed_cmp_csv = output_dir / f"reputation_status_scaling_seed_comparison_{args.mode}.csv"
    write_csv(run_csv, [asdict(r) for r in run_records])
    write_csv(agg_csv, [asdict(r) for r in aggregate_rows])
    write_csv(seed_cmp_csv, [asdict(r) for r in seed_comparison_rows])

    plot_kappa_heatmap(
        aggregate_rows,
        field="mean_tail_welfare",
        title="Mean tail welfare",
        output_file=output_dir / f"reputation_status_scaling_tail_welfare_{args.mode}.png",
    )
    plot_kappa_heatmap(
        aggregate_rows,
        field="mean_consensus_step_first",
        title="First Consensus Step",
        output_file=output_dir / f"mean_first_consensus_{args.mode}.png",
    )
    plot_kappa_heatmap(
        aggregate_rows,
        field="mean_consensus_step_final",
        title="Final Consensus Step",
        output_file=output_dir / f"mean_final_consensus_{args.mode}.png",
    )
    plot_kappa_heatmap(
        aggregate_rows,
        field="mean_num_consensus",
        title="Num Consensus'",
        output_file=output_dir / f"mean_num_consensus_{args.mode}.png",
    )

    if args.plot_leader_actor_rate != "off":
            layout = args.plot_leader_actor_rate
            sample_interval = int(args.plot_sample_interval)
            if layout in ("per_cell", "all"):
                plot_leader_actor_rate_per_cell(run_records, output_dir, args.mode, sample_interval)
            if layout in ("grid", "all"):
                plot_leader_actor_rate_grid(
                    run_records, output_dir, args.mode, sample_interval,
                    show_seeds=bool(args.plot_leader_actor_rate_grid_show_seeds),
                )
            if layout in ("overlay", "all"):
                plot_leader_actor_rate_overlay(run_records, output_dir, args.mode, sample_interval)

    generate_expC_report_figures(run_records, aggregate_rows, output_dir)
    
    print(f"\nWrote per-run CSV: {run_csv}")
    print(f"Wrote aggregate CSV: {agg_csv}")
    print(f"Wrote plots to: {output_dir}")
    print(f"Wrote seed-comparison CSV: {seed_cmp_csv}")


if __name__ == "__main__":
    main()
