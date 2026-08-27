"""
Experiment A: Personal Utility Baseline (γ = 0, κ = 0)

This script runs the baseline experiment where agents optimize
purely for personal utility without any social incentives.

Key properties:
- No reputation influence (gamma = 0)
- No status incentives (kappa = 0)
- Agents learn independently via reinforcement learning

Expected outcome:
- No follower structure
- No opinion leader
- Decentralized behavior across all agents

This serves as a control experiment for later comparisons.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.code_debugged import MultiAgentSystem, SystemConfig  # noqa: E402


# Output records
@dataclass
class RunRecord:
    mode: str
    reward_model: str
    num_states: int
    seed: int

    leader_id: int
    final_top_followers: int
    time_to_50pct_followers: int
    time_to_90pct_followers: int
    leader_switches: int
    tail_welfare: float

    final_pu: int
    final_rep: int
    final_status: int

    leader_role_final: str
    final_leader_is_pu: int
    final_leader_is_rep: int
    final_leader_is_status: int

    tail_top_follower_share: float


@dataclass
class AggregateRecord:
    mode: str
    reward_model: str
    num_states: int
    n_runs: int

    mean_final_top_followers: float
    std_final_top_followers: float
    ci95_final_top_followers: float

    mean_time_to_50pct_followers: float
    std_time_to_50pct_followers: float
    ci95_time_to_50pct_followers: float

    mean_time_to_90pct_followers: float
    std_time_to_90pct_followers: float
    ci95_time_to_90pct_followers: float

    mean_leader_switches: float
    std_leader_switches: float
    ci95_leader_switches: float

    mean_tail_welfare: float
    std_tail_welfare: float
    ci95_tail_welfare: float

    mean_final_pu: float
    std_final_pu: float
    ci95_final_pu: float

    mean_final_rep: float
    std_final_rep: float
    ci95_final_rep: float

    mean_final_status: float
    std_final_status: float
    ci95_final_status: float

    mean_tail_top_follower_share: float
    std_tail_top_follower_share: float
    ci95_tail_top_follower_share: float


# Command-line configuration
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure personal-utility scaling harness (Experiment A).")
    parser.add_argument("--mode", choices=["static", "async"], required=True)

    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--num-states-list", type=str, default="10")
    parser.add_argument("--num-actions", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=50000)

    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (inclusive).")
    parser.add_argument(
        "--selected-seeds",
        type=str,
        default="",
        help='Optional explicit comma-separated seed list (e.g. "0,1,2"). Overrides --seeds/--seed-start.',
    )

    parser.add_argument(
        "--reward-models",
        type=str,
        default="shared_base_gaussian",
        help='Comma-separated reward models, e.g. "shared_base_gaussian,simple_preferred_action".',
    )
    parser.add_argument("--reward-base-mu", type=float, default=0.5)
    parser.add_argument("--reward-base-sigma", type=float, default=0.15)
    parser.add_argument("--reward-agent-sigma", type=float, default=0.08)
    parser.add_argument("--reward-clip-min", type=float, default=0.01)
    parser.add_argument("--reward-clip-max", type=float, default=2.5)

    parser.add_argument("--delta", type=float, default=1e-6)

    parser.add_argument(
        "--actor-rate-driver-mode",
        choices=["standard", "status_if_followers_kappa0"],
        default="standard",
    )
    parser.add_argument(
        "--actor-rate-status-override-min-followers",
        type=int,
        default=10,
    )

    parser.add_argument("--tail-window", type=int, default=500)
    parser.add_argument("--role-update-s0", type=int, default=0)
    parser.add_argument("--role-update-T-seq", type=str, default="")
    parser.add_argument("--role-update-base-interval", type=int, default=3000)
    parser.add_argument(
        "--fixed-role-update-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--role-update-epochs", type=str, default="")
    parser.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    parser.add_argument(
        "--force-all-active-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--initial-actor-rate", type=float, default=0.7)
    parser.add_argument("--initial-participant-rate", type=float, default=0.7)
    parser.add_argument(
        "--eq9-averaging-mode",
        choices=["participants_only", "all_agents"],
        default="participants_only",
    )
    parser.add_argument(
        "--leader-update-mode",
        choices=["participants_only_post_eq9", "all_agents_post_eq9", "participants_only_pre_eq9"],
        default="participants_only_post_eq9",
    )
    parser.add_argument(
        "--numpy-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--c-threshold", type=float, default=0.1)
    parser.add_argument("--B-R", dest="B_R", type=float, default=0.8)
    parser.add_argument("--B-F", dest="B_F", type=float, default=0.6)

    parser.add_argument(
        "--trace-seeds",
        type=str,
        default="",
        help='Optional seeds for which to write sampled top-follower progression, e.g. "0,5".',
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=100,
        help="Downsample top-follower progression by recording every N timesteps.",
    )

    parser.add_argument(
        "--async-role-update-prob",
        type=float,
        default=None,
        help="Optional per-step Bernoulli probability for async subset role updates. "
             "If omitted, async uses independent per-agent clocks.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    return parser.parse_args()


# Small parsing helpers
def parse_csv_ints(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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
    selected = parse_csv_ints(getattr(args, "selected_seeds", ""))
    if selected:
        return sorted(set(seed for seed in selected if seed >= 0))
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))


def resolve_trace_seeds(args: argparse.Namespace) -> List[int]:
    return sorted(set(seed for seed in parse_csv_ints(getattr(args, "trace_seeds", "")) if seed >= 0))


def _role_to_label(role) -> str:
    if hasattr(role, "value"):
        return str(role.value).lower()
    if hasattr(role, "name"):
        return str(role.name).lower()
    return str(role).lower()


# System configuration for Experiment A
def make_config(args: argparse.Namespace, num_states: int, reward_model: str, mode: str) -> SystemConfig:
    role_interval = int(args.role_update_base_interval)
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)

    if mode == "async":
        role_interval = int(args.num_steps) + 1_000_000
        role_s0 = 0
        role_t_seq = []
        role_epochs = []

    return SystemConfig(
        num_agents=int(args.num_agents),
        num_states=int(num_states),
        num_actions=int(args.num_actions),
        num_time_steps=int(args.num_steps),
        M=1.0,
        u_0=0.1,
        actor_rate_driver_mode=str(args.actor_rate_driver_mode),
        actor_rate_status_override_min_followers=int(args.actor_rate_status_override_min_followers),

        gamma=0.0,
        kappa=0.0,

        c_threshold=float(args.c_threshold),
        B_R=float(args.B_R),
        B_F=float(args.B_F),
        delta=float(args.delta),

        eq9_averaging_mode=str(args.eq9_averaging_mode),
        leader_update_mode=str(args.leader_update_mode),

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
        force_all_active_debug=bool(getattr(args, "force_all_active_debug", False)),

        initial_actor_interaction_rate=float(args.initial_actor_rate),
        initial_participant_interaction_rate=float(args.initial_participant_rate),

        reward_model=str(reward_model),
        reward_base_mu=float(args.reward_base_mu),
        reward_base_sigma=float(args.reward_base_sigma),
        reward_agent_sigma=float(args.reward_agent_sigma),
        reward_clip_min=float(args.reward_clip_min),
        reward_clip_max=float(args.reward_clip_max),
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


def _tail_top_follower_share(follower_counts: np.ndarray, tail_window: int, denom: int) -> float:
    if follower_counts.size == 0:
        return 0.0
    tail_window = min(int(tail_window), follower_counts.shape[0])
    tail = follower_counts[-tail_window:]
    return float(np.mean(np.max(tail, axis=1) / max(1, denom)))


def _sample_progression_rows(
    follower_counts: np.ndarray,
    sample_every: int,
    seed: int,
    mode: str,
    reward_model: str,
    num_states: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if follower_counts.size == 0:
        return out

    sample_every = max(1, int(sample_every))
    top_series = np.max(follower_counts, axis=1)

    for idx in range(0, len(top_series), sample_every):
        out.append(
            {
                "mode": mode,
                "reward_model": reward_model,
                "num_states": int(num_states),
                "seed": int(seed),
                "t": int(idx + 1),
                "top_followers": int(top_series[idx]),
            }
        )
    if (len(top_series) - 1) % sample_every != 0:
        out.append(
            {
                "mode": mode,
                "reward_model": reward_model,
                "num_states": int(num_states),
                "seed": int(seed),
                "t": int(len(top_series)),
                "top_followers": int(top_series[-1]),
            }
        )
    return out


# Single simulation run
def run_single(
    args: argparse.Namespace,
    mode: str,
    reward_model: str,
    num_states: int,
    seed: int,
    collect_trace: bool = False,
) -> Tuple[RunRecord, List[Dict[str, object]], List[Dict[str, object]]]:
    np.random.seed(seed)
    config = make_config(args, num_states=num_states, reward_model=reward_model, mode=mode)
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

    threshold_50 = int(np.ceil(0.50 * (args.num_agents - 1)))
    threshold_90 = int(np.ceil(0.90 * (args.num_agents - 1)))

    time_to_50 = _time_to_threshold(top_follower_series, threshold_50)
    time_to_90 = _time_to_threshold(top_follower_series, threshold_90)
    leader_switches = _leader_switches(leader_series)

    # Tail welfare is averaged over the final tail_window timesteps.
    tail_window = min(int(args.tail_window), len(social_welfare)) if len(social_welfare) > 0 else 0
    tail_welfare = float(np.mean(social_welfare[-tail_window:])) if tail_window > 0 else float("nan")

    leader_id = int(results["opinion_leader"])
    final_roles = [_role_to_label(r) for r in results["final_roles"]]
    leader_role_final = final_roles[leader_id] if leader_id >= 0 else "none"

    final_pu = sum(1 for r in final_roles if r == "personal_utility")
    final_rep = sum(1 for r in final_roles if r == "reputation")
    final_status = sum(1 for r in final_roles if r == "status")

    tail_share = _tail_top_follower_share(
        follower_counts=follower_counts,
        tail_window=args.tail_window,
        denom=max(1, args.num_agents - 1),
    )

    run_record = RunRecord(
        mode=str(mode),
        reward_model=str(reward_model),
        num_states=int(num_states),
        seed=int(seed),

        leader_id=int(leader_id),
        final_top_followers=int(max(results["final_followers"])),
        time_to_50pct_followers=int(time_to_50),
        time_to_90pct_followers=int(time_to_90),
        leader_switches=int(leader_switches),
        tail_welfare=float(tail_welfare),

        final_pu=int(final_pu),
        final_rep=int(final_rep),
        final_status=int(final_status),

        leader_role_final=str(leader_role_final),
        final_leader_is_pu=int(leader_role_final == "personal_utility"),
        final_leader_is_rep=int(leader_role_final == "reputation"),
        final_leader_is_status=int(leader_role_final == "status"),

        tail_top_follower_share=float(tail_share),
    )

    progression_rows: List[Dict[str, object]] = []
    if collect_trace:
        progression_rows = _sample_progression_rows(
            follower_counts=follower_counts,
            sample_every=int(args.trace_every),
            seed=int(seed),
            mode=str(mode),
            reward_model=str(reward_model),
            num_states=int(num_states),
        )

    agent_trace_rows: List[Dict[str, object]] = []
    if collect_trace:
        final_est_pu = results.get("estimated_reward_pu_history", [])
        final_est_rep = results.get("estimated_reward_rep_history", [])
        final_est_status = results.get("estimated_reward_status_history", [])
        final_actor_rates = results.get("actor_interaction_rate_history", [])

        last_est_pu = final_est_pu[-1] if len(final_est_pu) > 0 else [float("nan")] * args.num_agents
        last_est_rep = final_est_rep[-1] if len(final_est_rep) > 0 else [float("nan")] * args.num_agents
        last_est_status = final_est_status[-1] if len(final_est_status) > 0 else [float("nan")] * args.num_agents
        last_actor_rates = final_actor_rates[-1] if len(final_actor_rates) > 0 else [float("nan")] * args.num_agents

        true_rep_snapshot = system._compute_true_reputation_vector()
        true_rep = np.asarray(true_rep_snapshot["true_reputation"], dtype=float)

        for agent_id in range(args.num_agents):
            agent_trace_rows.append(
                {
                    "mode": str(mode),
                    "reward_model": str(reward_model),
                    "num_states": int(num_states),
                    "seed": int(seed),
                    "agent_id": int(agent_id),
                    "final_role": final_roles[agent_id],
                    "followers": int(results["final_followers"][agent_id]),
                    "estimated_pu": float(last_est_pu[agent_id]),
                    "estimated_rep": float(last_est_rep[agent_id]),
                    "estimated_status": float(last_est_status[agent_id]),
                    "actor_rate": float(last_actor_rates[agent_id]),
                    "true_reputation": float(true_rep[agent_id]),
                }
            )

    return run_record, progression_rows, agent_trace_rows


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
    grouped: Dict[Tuple[str, str, int], List[RunRecord]] = {}
    for r in records:
        grouped.setdefault((r.mode, r.reward_model, r.num_states), []).append(r)

    rows: List[AggregateRecord] = []
    for (mode, reward_model, num_states), recs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        m1, s1, c1 = _mean_std_ci([r.final_top_followers for r in recs])
        m2, s2, c2 = _mean_std_ci([r.time_to_50pct_followers for r in recs if r.time_to_50pct_followers >= 0])
        m3, s3, c3 = _mean_std_ci([r.time_to_90pct_followers for r in recs if r.time_to_90pct_followers >= 0])
        m4, s4, c4 = _mean_std_ci([r.leader_switches for r in recs])
        m5, s5, c5 = _mean_std_ci([r.tail_welfare for r in recs])
        m6, s6, c6 = _mean_std_ci([r.final_pu for r in recs])
        m7, s7, c7 = _mean_std_ci([r.final_rep for r in recs])
        m8, s8, c8 = _mean_std_ci([r.final_status for r in recs])
        m9, s9, c9 = _mean_std_ci([r.tail_top_follower_share for r in recs])

        rows.append(
            AggregateRecord(
                mode=mode,
                reward_model=reward_model,
                num_states=int(num_states),
                n_runs=len(recs),

                mean_final_top_followers=m1,
                std_final_top_followers=s1,
                ci95_final_top_followers=c1,

                mean_time_to_50pct_followers=m2,
                std_time_to_50pct_followers=s2,
                ci95_time_to_50pct_followers=c2,

                mean_time_to_90pct_followers=m3,
                std_time_to_90pct_followers=s3,
                ci95_time_to_90pct_followers=c3,

                mean_leader_switches=m4,
                std_leader_switches=s4,
                ci95_leader_switches=c4,

                mean_tail_welfare=m5,
                std_tail_welfare=s5,
                ci95_tail_welfare=c5,

                mean_final_pu=m6,
                std_final_pu=s6,
                ci95_final_pu=c6,

                mean_final_rep=m7,
                std_final_rep=s7,
                ci95_final_rep=c7,

                mean_final_status=m8,
                std_final_status=s8,
                ci95_final_status=c8,

                mean_tail_top_follower_share=m9,
                std_tail_top_follower_share=s9,
                ci95_tail_top_follower_share=c9,
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
def plot_metric(
    aggregate_rows: Sequence[AggregateRecord],
    x_field: str,
    y_field: str,
    ylabel: str,
    output_file: Path,
    title: str | None = None,
) -> None:
    if not aggregate_rows:
        return

    groups: Dict[Tuple[str, str], List[AggregateRecord]] = {}
    for row in aggregate_rows:
        groups.setdefault((row.mode, row.reward_model), []).append(row)

    plt.figure(figsize=(6.8, 4.5))
    for (mode, reward_model), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: getattr(r, x_field))
        xs = np.array([getattr(r, x_field) for r in rows], dtype=float)
        ys = np.array([getattr(r, y_field) for r in rows], dtype=float)
        plt.plot(xs, ys, "-o", linewidth=1.8, label=f"{mode} | {reward_model}")

    plt.xlabel(x_field.replace("_", " "))
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.25)
    if len(groups) > 1:
        plt.legend()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def plot_progression(
    progression_rows: Sequence[Dict[str, object]],
    output_file: Path,
    reward_model: str,
    num_states: int,
    mode: str,
) -> None:
    if not progression_rows:
        return

    rows = [
        r for r in progression_rows
        if r["reward_model"] == reward_model and int(r["num_states"]) == int(num_states) and r["mode"] == mode
    ]
    if not rows:
        return

    rows = sorted(rows, key=lambda r: (int(r["seed"]), int(r["t"])))
    seeds = sorted(set(int(r["seed"]) for r in rows))

    plt.figure(figsize=(7.2, 4.6))
    for seed in seeds:
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        xs = [int(r["t"]) for r in seed_rows]
        ys = [int(r["top_followers"]) for r in seed_rows]
        plt.plot(xs, ys, linewidth=1.5, alpha=0.9, label=f"seed {seed}")

    plt.xlabel("timestep")
    plt.ylabel("top followers")
    plt.title(f"Experiment A progression | {mode} | {reward_model} | states={num_states}")
    plt.grid(alpha=0.25)
    if len(seeds) <= 10:
        plt.legend(fontsize=8, ncol=2)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def build_seed_comparison_rows(records: Sequence[RunRecord]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in sorted(records, key=lambda x: (x.reward_model, x.num_states, x.seed, x.mode)):
        out.append(
            {
                "mode": r.mode,
                "reward_model": r.reward_model,
                "num_states": r.num_states,
                "seed": r.seed,
                "leader_id": r.leader_id,
                "leader_role_final": r.leader_role_final,
                "final_top_followers": r.final_top_followers,
                "time_to_50pct_followers": r.time_to_50pct_followers,
                "time_to_90pct_followers": r.time_to_90pct_followers,
                "leader_switches": r.leader_switches,
                "tail_welfare": r.tail_welfare,
                "final_pu": r.final_pu,
                "final_rep": r.final_rep,
                "final_status": r.final_status,
                "tail_top_follower_share": r.tail_top_follower_share,
            }
        )
    return out


def generate_expA_report_figure(progression_rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    """
    Generate the final report figure for Experiment A.
    This figure shows that the maximum follower count remains zero over time
    when γ = 0 and κ = 0. It is used in Section 5.1 of the report.
    """

    if not progression_rows:
        return

    df_rows = list(progression_rows)
    ts: Dict[int, List[int]] = {}

    for row in df_rows:
        t = int(row["t"])
        ts.setdefault(t, []).append(int(row["top_followers"]))

    xs = sorted(ts.keys())
    means = np.array([np.mean(ts[t]) for t in xs], dtype=float)
    stds = np.array([np.std(ts[t], ddof=1) if len(ts[t]) >= 2 else 0.0 for t in xs], dtype=float)
    counts = np.array([len(ts[t]) for t in xs], dtype=float)
    ci95 = 1.96 * stds / np.sqrt(np.maximum(counts, 1))

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(xs, means, linewidth=2.5)

    if np.max(ci95) > 0:
        plt.fill_between(xs, means - ci95, means + ci95, alpha=0.2)
    
    plt.axhline(0, linestyle="--", linewidth=1.2)
    plt.title("No follower structure emerges ($\\gamma=0, \\kappa=0$)")
    plt.xlabel("Timestep")
    plt.ylabel("Maximum number of followers")
    plt.ylim(-0.5, 1)
    plt.grid(alpha=0.25)

    fig_dir = output_dir.parent / "final_report_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "expA_followers_timeseries.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Wrote report figure: {out_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_models = parse_csv_strs(args.reward_models)
    num_states_list = parse_csv_ints(args.num_states_list)
    seeds = resolve_seeds(args)
    trace_seeds = resolve_trace_seeds(args)

    run_records: List[RunRecord] = []
    progression_rows: List[Dict[str, object]] = []
    agent_trace_rows: List[Dict[str, object]] = []

    for reward_model in reward_models:
        for num_states in num_states_list:
            for seed in seeds:
                record, prog_rows, trace_rows = run_single(
                    args=args,
                    mode=args.mode,
                    reward_model=reward_model,
                    num_states=num_states,
                    seed=seed,
                    collect_trace=(seed in trace_seeds),
                )
                run_records.append(record)
                progression_rows.extend(prog_rows)
                agent_trace_rows.extend(trace_rows)

                print(
                    f"[done] mode={record.mode} reward={record.reward_model} states={record.num_states} "
                    f"seed={record.seed} leader={record.leader_id} top_followers={record.final_top_followers} "
                    f"role={record.leader_role_final} final_pu={record.final_pu} "
                    f"final_rep={record.final_rep} final_status={record.final_status} "
                    f"tail_welfare={record.tail_welfare:.4f}"
                )

    aggregate_rows = aggregate(run_records)

    run_csv = output_dir / f"pu_scaling_runs_{args.mode}.csv"
    agg_csv = output_dir / f"pu_scaling_aggregate_{args.mode}.csv"
    seed_compare_csv = output_dir / f"pu_scaling_seed_comparison_{args.mode}.csv"
    progression_csv = output_dir / f"pu_progression_{args.mode}.csv"
    agent_trace_csv = output_dir / f"pu_agent_traces_{args.mode}.csv"

    write_csv(run_csv, [asdict(r) for r in run_records])
    write_csv(agg_csv, [asdict(r) for r in aggregate_rows])
    write_csv(seed_compare_csv, build_seed_comparison_rows(run_records))
    write_csv(progression_csv, progression_rows)
    write_csv(agent_trace_csv, agent_trace_rows)

    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_final_top_followers",
        ylabel="Mean final top followers",
        output_file=output_dir / f"pu_scaling_final_top_followers_{args.mode}.png",
        title="Experiment A: final top followers",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_time_to_50pct_followers",
        ylabel="Mean time to 50% followers",
        output_file=output_dir / f"pu_scaling_time_to_50pct_{args.mode}.png",
        title="Experiment A: time to 50% followers",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_time_to_90pct_followers",
        ylabel="Mean time to 90% followers",
        output_file=output_dir / f"pu_scaling_time_to_90pct_{args.mode}.png",
        title="Experiment A: time to 90% followers",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_leader_switches",
        ylabel="Mean leader switches",
        output_file=output_dir / f"pu_scaling_leader_switches_{args.mode}.png",
        title="Experiment A: leader switches",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_tail_welfare",
        ylabel="Mean tail welfare",
        output_file=output_dir / f"pu_scaling_tail_welfare_{args.mode}.png",
        title="Experiment A: tail welfare",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_final_pu",
        ylabel="Mean final PU count",
        output_file=output_dir / f"pu_scaling_final_pu_{args.mode}.png",
        title="Experiment A: final PU count",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_final_rep",
        ylabel="Mean final REP count",
        output_file=output_dir / f"pu_scaling_final_rep_{args.mode}.png",
        title="Experiment A: final REP count",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_final_status",
        ylabel="Mean final STATUS count",
        output_file=output_dir / f"pu_scaling_final_status_{args.mode}.png",
        title="Experiment A: final STATUS count",
    )
    plot_metric(
        aggregate_rows=aggregate_rows,
        x_field="num_states",
        y_field="mean_tail_top_follower_share",
        ylabel="Mean tail top-follower share",
        output_file=output_dir / f"pu_scaling_tail_top_follower_share_{args.mode}.png",
        title="Experiment A: tail top-follower share",
    )

    for reward_model in reward_models:
        for num_states in num_states_list:
            plot_progression(
                progression_rows=progression_rows,
                output_file=output_dir / f"pu_progression_{args.mode}_{reward_model}_S{num_states}.png",
                reward_model=reward_model,
                num_states=num_states,
                mode=args.mode,
            )

    generate_expA_report_figure(progression_rows, output_dir)

    print(f"\nWrote per-run CSV: {run_csv}")
    print(f"Wrote aggregate CSV: {agg_csv}")
    print(f"Wrote seed-comparison CSV: {seed_compare_csv}")
    if progression_rows:
        print(f"Wrote progression CSV: {progression_csv}")
    if agent_trace_rows:
        print(f"Wrote agent-trace CSV: {agent_trace_csv}")
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()