"""
Experiment C: Status Scaling (kappa sweep with fixed gamma)

This script runs the main Experiment C in the paper:
- gamma is fixed (reputation strength)
- kappa is varied (status incentive strength)

Goals:
1. Study how status incentives affect leader formation
2. Measure whether status leads to welfare improvements
3. Check whether emergent norms are welfare-optimal

This experiment evaluates whether adding status incentives 
helps or harms collective welfare compared to reputation-only.
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

@dataclass
class AggregateRecord:
    mode: str
    gamma: float
    kappa: float
    n_runs: int
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
    parser.add_argument("--gammas", type=str, default="0,1,2,3,4")
    parser.add_argument("--kappas", type=str, default="0,0.01,0.02,0.05,0.1")
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
    return parser.parse_args()


# Small parsing helpers
def parse_gammas(gamma_text: str) -> List[float]:
    parts = [p.strip() for p in gamma_text.split(",") if p.strip()]
    return [float(x) for x in parts]

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
        gamma=gamma,
        kappa=kappa,
        c_threshold=0.1,
        B_R=0.3,
        B_F=0.2,
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
            results = _finalize_results(system)
    else:
        with redirect_stdout(io.StringIO()):
            for _ in range(args.num_steps):
                system.step()
            results = _finalize_results(system)

    follower_counts = np.asarray(results["follower_counts"], dtype=float)
    role_history = np.asarray(results.get("role_label_history", []), dtype=object)
    social_welfare = np.asarray(results.get("paper_welfare_followers_only", []), dtype=float)

    top_follower_series = follower_counts.max(axis=1)
    leader_series = _leader_series_from_follower_counts(follower_counts)
    threshold_90 = int(np.ceil(0.90 * (args.num_agents - 1)))
    time_to_90 = _time_to_threshold(top_follower_series, threshold_90)
    leader_switches = _leader_switches(leader_series)

    tail_window = min(int(args.tail_window), len(social_welfare)) if len(social_welfare) > 0 else 0
    tail_welfare = float(np.mean(social_welfare[-tail_window:])) if tail_window > 0 else float("nan")

    leader_id = int(results["opinion_leader"])
    final_roles = [_role_to_label(r) for r in results["final_roles"]]
    leader_role_final = final_roles[leader_id] if leader_id >= 0 else "none"
    leader_is_status_final = int(leader_role_final == "status")
    final_status_count = sum(1 for r in final_roles if r == "status")
    tail_status_leader_share = _tail_status_leader_share(role_history, leader_series, tail_window)
    tail_status_agent_share = _tail_status_agent_share(role_history, tail_window)

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
        m2, s2, c2 = _mean_std_ci([r.time_to_90pct_followers for r in recs if r.time_to_90pct_followers >= 0])
        m3, s3, c3 = _mean_std_ci([r.leader_switches for r in recs])
        m4, s4, c4 = _mean_std_ci([r.tail_welfare for r in recs])
        m5, s5, c5 = _mean_std_ci([r.leader_is_status_final for r in recs])
        m6, s6, c6 = _mean_std_ci([r.final_status_count for r in recs])
        m7, s7, c7 = _mean_std_ci([r.tail_status_leader_share for r in recs])
        m8, s8, c8 = _mean_std_ci([r.tail_status_agent_share for r in recs])

        
        m9, s9, c9 = _mean_std_ci([r.final_norm_welfare_check for r in recs if np.isfinite(r.final_norm_welfare_check)])
        m10, s10, c10 = _mean_std_ci([r.best_norm_welfare for r in recs if np.isfinite(r.best_norm_welfare)])
        m11, s11, c11 = _mean_std_ci([r.welfare_gap_to_best for r in recs if np.isfinite(r.welfare_gap_to_best)])
        m12, s12, c12 = _mean_std_ci([r.is_final_norm_optimal for r in recs if r.is_final_norm_optimal >= 0])
        rows.append(
            AggregateRecord(
                mode=mode,
                gamma=float(gamma),
                kappa=float(kappa),
                n_runs=len(recs),
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
    """
    Mean `field` vs kappa, one line per gamma. With a single gamma this looks
    identical to the original plot; with several it stops silently overplotting
    every gamma's points onto one misleading line.
    """
    gammas = sorted({float(r.gamma) for r in aggregate_rows})
    plt.figure(figsize=(6.4, 4.2))
    cmap = plt.get_cmap("viridis")
    for idx, gamma in enumerate(gammas):
        rows = sorted((r for r in aggregate_rows if float(r.gamma) == gamma), key=lambda r: r.kappa)
        if not rows:
            continue
        kappas = np.array([r.kappa for r in rows], dtype=float)
        means = np.array([getattr(r, field) for r in rows], dtype=float)
        color = cmap(idx / max(1, len(gammas) - 1)) if len(gammas) > 1 else None
        plt.plot(kappas, means, "-o", linewidth=1.8, color=color, label=f"gamma={gamma:g}")
    plt.xlabel("kappa")
    plt.ylabel(ylabel)
    if len(gammas) > 1:
        plt.legend(fontsize=8, title="gamma")
    plt.grid(alpha=0.25)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()

    # Companion heatmap: only meaningful once both axes actually vary.
    kappas_all = sorted({float(r.kappa) for r in aggregate_rows})
    if len(gammas) > 1 and len(kappas_all) > 1:
        heatmap_path = output_file.with_name(output_file.stem + "_heatmap" + output_file.suffix)
        plot_gamma_kappa_heatmap(
            aggregate_rows, field=field, title=ylabel, output_file=heatmap_path,
        )


def plot_gamma_kappa_heatmap(
    aggregate_rows: Sequence[AggregateRecord],
    field: str,
    title: str,
    output_file: Path,
    fmt: str = "{:.3g}",
) -> None:
    """gamma (x-axis) x kappa (y-axis) heatmap of one aggregate field."""
    gammas = sorted({float(r.gamma) for r in aggregate_rows})
    kappas = sorted({float(r.kappa) for r in aggregate_rows})
    if not gammas or not kappas:
        return

    lookup = {(float(r.gamma), float(r.kappa)): getattr(r, field) for r in aggregate_rows}
    grid = np.full((len(kappas), len(gammas)), np.nan, dtype=float)
    for j, g in enumerate(gammas):
        for i, k in enumerate(kappas):
            val = lookup.get((g, k))
            if val is not None and np.isfinite(val):
                grid[i, j] = float(val)

    fig, ax = plt.subplots(figsize=(1.1 * len(gammas) + 2.6, 0.9 * len(kappas) + 2.2))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f"{g:g}" for g in gammas])
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:g}" for k in kappas])
    ax.set_xlabel("gamma")
    ax.set_ylabel("kappa")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(kappas)):
        for j in range(len(gammas)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
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
    """
    One errorbar line per gamma (if the "gamma" column has more than one
    value), x=kappa. With a single gamma this is identical to the original
    single-line plot.
    """
    plt.figure(figsize=(7, 4.6))
    if "gamma" in df.columns and df["gamma"].nunique() > 1:
        for gamma, sub_df in df.groupby("gamma"):
            sub_df = sub_df.sort_values(x)
            plt.errorbar(
                sub_df[x], sub_df[y], yerr=sub_df[yerr],
                marker="o", capsize=4, linewidth=1.8, label=f"gamma={gamma:g}",
            )
        plt.legend(fontsize=8, title="gamma")
    else:
        df = df.sort_values(x)
        plt.errorbar(df[x], df[y], yerr=df[yerr], marker="o", capsize=4, linewidth=1.8)
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


def generate_expC_report_figures(
    run_records: Sequence[RunRecord],
    aggregate_rows: Sequence[AggregateRecord],
    output_dir: Path,
) -> None:
    """
    Generate final report figures for Experiment C.

    Outputs:
    - expC_leader_is_status_vs_kappa.png
    - expC_tail_welfare_vs_kappa.png
    - expC_welfare_gap_vs_kappa.png
    - expC_optimal_norm_probability_vs_kappa.png
    """

    fig_dir = output_dir.parent / "final_report_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    runs_df = pd.DataFrame([asdict(r) for r in run_records])
    agg_df = pd.DataFrame([asdict(r) for r in aggregate_rows])

    agg_df = agg_df.sort_values("kappa")
    runs_df = runs_df.sort_values("kappa")

    # C1: leader is status
    _errorbar_plot(
        df=agg_df,
        x="kappa",
        y=_find_col(agg_df, ["mean_leader_is_status_final"]),
        yerr=_find_col(agg_df, ["ci95_leader_is_status_final"]),
        title="Final leader is STATUS vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Probability final leader is STATUS",
        out_path=fig_dir / "expC_leader_is_status_vs_kappa.png",
        ylim=(-0.05, 1.05),
    )

    # C2: tail welfare
    _errorbar_plot(
        df=agg_df,
        x="kappa",
        y=_find_col(agg_df, ["mean_tail_welfare"]),
        yerr=_find_col(agg_df, ["ci95_tail_welfare"]),
        title="Tail welfare vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Mean tail welfare",
        out_path=fig_dir / "expC_tail_welfare_vs_kappa.png",
    )

    if "mean_welfare_gap_to_best" in agg_df.columns:
        gap_df = agg_df
        y = "mean_welfare_gap_to_best"
        ci = "ci95_welfare_gap_to_best"
    else:
        grouped = runs_df.groupby("kappa")["welfare_gap_to_best"]
        gap_df = grouped.agg(["mean", "std", "count"]).reset_index()
        gap_df["ci95"] = 1.96 * gap_df["std"].fillna(0) / np.sqrt(gap_df["count"])
        y = "mean"
        ci = "ci95"

    _errorbar_plot(
        df=gap_df,
        x="kappa",
        y=y,
        yerr=ci,
        title="Welfare gap to best norm vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Mean welfare gap",
        out_path=fig_dir / "expC_welfare_gap_vs_kappa.png",
    )

    # C4: optimal norm probability
    if "mean_is_final_norm_optimal" in agg_df.columns:
        opt_df = agg_df
        y = "mean_is_final_norm_optimal"
        ci = "ci95_is_final_norm_optimal"
    else:
        grouped = runs_df.groupby("kappa")["is_final_norm_optimal"]
        opt_df = grouped.agg(["mean", "std", "count"]).reset_index()
        opt_df["ci95"] = 1.96 * opt_df["std"].fillna(0) / np.sqrt(opt_df["count"])
        y = "mean"
        ci = "ci95"

    _errorbar_plot(
        df=opt_df,
        x="kappa",
        y=y,
        yerr=ci,
        title="Probability final norm is optimal vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Probability optimal",
        out_path=fig_dir / "expC_optimal_norm_probability_vs_kappa.png",
        ylim=(-0.05, 1.05),
    )

    # Heatmaps: the errorbar plots above are clearest for trends across kappa;
    # a heatmap shows the full gamma x kappa surface at a glance. Only
    # meaningful once both axes actually vary.
    if agg_df["gamma"].nunique() > 1 and agg_df["kappa"].nunique() > 1:
        plot_gamma_kappa_heatmap(
            aggregate_rows, field="mean_leader_is_status_final",
            title="P(final leader is STATUS)",
            output_file=fig_dir / "expC_heatmap_leader_is_status.png",
        )
        plot_gamma_kappa_heatmap(
            aggregate_rows, field="mean_tail_welfare",
            title="Mean tail welfare",
            output_file=fig_dir / "expC_heatmap_tail_welfare.png",
        )
        plot_gamma_kappa_heatmap(
            aggregate_rows, field="mean_welfare_gap_to_best",
            title="Mean welfare gap to best norm",
            output_file=fig_dir / "expC_heatmap_welfare_gap.png",
        )
        plot_gamma_kappa_heatmap(
            aggregate_rows, field="mean_is_final_norm_optimal",
            title="P(final norm is optimal)",
            output_file=fig_dir / "expC_heatmap_optimal_norm.png",
        )

    print(f"[\u2713] Report figures saved to {fig_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gammas = parse_gammas(args.gammas)
    kappas = parse_kappas(args.kappas)
    seeds = resolve_seeds(args)

    run_records: List[RunRecord] = []
    for gamma in gammas:
        for kappa in kappas:
            for seed in seeds:
                record = run_single(args, mode=args.mode, gamma=gamma, kappa=kappa, seed=seed)
                run_records.append(record)
                
                print(
                    f"[done] mode={record.mode} gamma={record.gamma:g} kappa={record.kappa:g} seed={record.seed} "
                    f"leader={record.leader_id} top_followers={record.final_top_followers} "
                    f"leader_role={record.leader_role_final} n_status={record.final_status_count} "
                    f"tail_welfare={record.tail_welfare:.4f} "
                    f"optimal={record.is_final_norm_optimal} gap={record.welfare_gap_to_best:.6f}"
                )

    aggregate_rows = aggregate(run_records)
    seed_comparison_rows = build_seed_comparison(run_records)

    run_csv = output_dir / f"status_scaling_runs_{args.mode}.csv"
    agg_csv = output_dir / f"status_scaling_aggregate_{args.mode}.csv"
    seed_cmp_csv = output_dir / f"status_scaling_seed_comparison_{args.mode}.csv"
    write_csv(run_csv, [asdict(r) for r in run_records])
    write_csv(agg_csv, [asdict(r) for r in aggregate_rows])
    write_csv(seed_cmp_csv, [asdict(r) for r in seed_comparison_rows])

    plot_metric(
        aggregate_rows,
        field="mean_final_top_followers",
        ylabel="Mean final top followers",
        output_file=output_dir / f"status_scaling_final_top_followers_{args.mode}.png",
    )
    plot_metric(
        aggregate_rows,
        field="mean_time_to_90pct_followers",
        ylabel="Mean time to 90% followers",
        output_file=output_dir / f"status_scaling_time_to_90pct_{args.mode}.png",
    )
    plot_metric(
        aggregate_rows,
        field="mean_leader_switches",
        ylabel="Mean leader switches",
        output_file=output_dir / f"status_scaling_leader_switches_{args.mode}.png",
    )
    plot_metric(
        aggregate_rows,
        field="mean_tail_welfare",
        ylabel="Mean tail welfare",
        output_file=output_dir / f"status_scaling_tail_welfare_{args.mode}.png",
    )
    plot_metric(
        aggregate_rows,
        field="mean_leader_is_status_final",
        ylabel="P(final leader is STATUS)",
        output_file=output_dir / f"status_scaling_leader_is_status_{args.mode}.png",
    )
    plot_metric(
        aggregate_rows,
        field="mean_final_status_count",
        ylabel="Mean final status count",
        output_file=output_dir / f"status_scaling_final_status_count_{args.mode}.png",
    )
    plot_metric(
        aggregate_rows,
        field="mean_tail_status_leader_share",
        ylabel="Tail share: top leader in STATUS",
        output_file=output_dir / f"status_scaling_tail_status_leader_share_{args.mode}.png",
    )
    plot_metric(
        aggregate_rows,
        field="mean_tail_status_agent_share",
        ylabel="Tail share: all agents in STATUS",
        output_file=output_dir / f"status_scaling_tail_status_agent_share_{args.mode}.png",
    )

    generate_expC_report_figures(run_records, aggregate_rows, output_dir)
    
    print(f"\nWrote per-run CSV: {run_csv}")
    print(f"Wrote aggregate CSV: {agg_csv}")
    print(f"Wrote plots to: {output_dir}")
    print(f"Wrote seed-comparison CSV: {seed_cmp_csv}")


if __name__ == "__main__":
    main()
