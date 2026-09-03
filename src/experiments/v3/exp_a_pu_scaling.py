"""
Experiment A: Personal Utility Baseline (gamma = 0, kappa = 0).

Control experiment: agents optimise purely for personal utility with no social
incentives, so no follower structure and no opinion leader should emerge.

This is the reference port. Its CSV output is byte-identical to
src/experiments/v2/pu_scaling.py for the same arguments -- see
tools/parity_check.py. 987 lines became ~230, of which ~90 are the record
schema and the figure list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.harness import metrics  # noqa: E402
from experiments.harness.aggregate import Triple, nonneg  # noqa: E402
from experiments.harness.axes import Axis  # noqa: E402
from experiments.harness.cli import (  # noqa: E402
    add_core_arguments,
    parse_csv_ints,
    parse_csv_strs,
)
from experiments.harness.experiment import Experiment  # noqa: E402
from experiments.harness.extras.tracking import (  # noqa: E402
    AgentTracePlugin,
    FollowerProgressionPlugin,
)
from experiments.harness.plotting import plot_mean_band, plot_metric, plot_series_by_group  # noqa: E402
from experiments.harness.plugins import MetricsPlugin, RunContext, SweepContext, SweepPlugin  # noqa: E402


# ------------------------------------------------------------------ CLI ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure personal-utility scaling harness (Experiment A)."
    )
    add_core_arguments(
        parser,
        defaults={
            # Experiment A's own defaults, which differ from B/C.
            "reward_base_sigma": 0.15,
            "reward_agent_sigma": 0.08,
            "B_R": 0.8,
            "B_F": 0.6,
            "delta": 1e-6,
            "num_agents": 100,
            "num_steps": 50000,
            "role_update_base_interval": 3000,
            "output_dir": str(Path(__file__).resolve().parent / "outputs"),
        },
    )
    group = parser.add_argument_group("experiment A sweep")
    group.add_argument("--num-states-list", type=str, default="10")
    group.add_argument(
        "--reward-models",
        type=str,
        default="shared_base_gaussian",
        help='Comma-separated reward models, e.g. "shared_base_gaussian,simple_preferred_action".',
    )
    return parser


def build_axes(args: argparse.Namespace) -> Sequence[Axis]:
    # Order matters: reward_model is the outer loop in the legacy harness, so it
    # comes first and the run order (hence CSV row order) is preserved.
    return (
        Axis.of("reward_model", parse_csv_strs(args.reward_models)),
        Axis.of("num_states", parse_csv_ints(args.num_states_list)),
    )


# -------------------------------------------------------------- metrics ---

CORE_COLUMNS = (
    "leader_id",
    "final_top_followers",
    "time_to_50pct_followers",
    "time_to_90pct_followers",
    "leader_switches",
    "tail_welfare",
    "final_pu",
    "final_rep",
    "final_status",
    "leader_role_final",
    "final_leader_is_pu",
    "final_leader_is_rep",
    "final_leader_is_status",
    "tail_top_follower_share",
)


def measure_core(ctx: RunContext) -> Dict[str, Any]:
    s = ctx.summary
    n = int(ctx.args.num_agents)

    threshold_50 = int(np.ceil(0.50 * (n - 1)))
    threshold_90 = int(np.ceil(0.90 * (n - 1)))

    tail_welfare = (
        float(np.mean(s.social_welfare[-s.tail_window:]))
        if s.tail_window > 0
        else float("nan")
    )

    leader_role = s.final_roles[s.leader_id] if s.leader_id >= 0 else "none"

    return {
        "leader_id": int(s.leader_id),
        "final_top_followers": int(max(s.final_followers)),
        "time_to_50pct_followers": int(
            metrics.time_to_threshold(s.top_follower_series, threshold_50)
        ),
        "time_to_90pct_followers": int(
            metrics.time_to_threshold(s.top_follower_series, threshold_90)
        ),
        "leader_switches": int(metrics.leader_switches(s.leader_series)),
        "tail_welfare": float(tail_welfare),
        "final_pu": sum(1 for r in s.final_roles if r == "personal_utility"),
        "final_rep": sum(1 for r in s.final_roles if r == "reputation"),
        "final_status": sum(1 for r in s.final_roles if r == "status"),
        "leader_role_final": str(leader_role),
        "final_leader_is_pu": int(leader_role == "personal_utility"),
        "final_leader_is_rep": int(leader_role == "reputation"),
        "final_leader_is_status": int(leader_role == "status"),
        "tail_top_follower_share": float(
            metrics.tail_top_follower_share_elementwise(
                s.follower_counts, ctx.args.tail_window, denom=max(1, n - 1)
            )
        ),
    }


# -------------------------------------------------------------- figures ---

METRIC_FIGURES = [
    ("mean_final_top_followers", "Mean final top followers", "final_top_followers",
     "Experiment A: final top followers"),
    ("mean_time_to_50pct_followers", "Mean time to 50% followers", "time_to_50pct",
     "Experiment A: time to 50% followers"),
    ("mean_time_to_90pct_followers", "Mean time to 90% followers", "time_to_90pct",
     "Experiment A: time to 90% followers"),
    ("mean_leader_switches", "Mean leader switches", "leader_switches",
     "Experiment A: leader switches"),
    ("mean_tail_welfare", "Mean tail welfare", "tail_welfare",
     "Experiment A: tail welfare"),
    ("mean_final_pu", "Mean final PU count", "final_pu", "Experiment A: final PU count"),
    ("mean_final_rep", "Mean final REP count", "final_rep", "Experiment A: final REP count"),
    ("mean_final_status", "Mean final STATUS count", "final_status",
     "Experiment A: final STATUS count"),
    ("mean_tail_top_follower_share", "Mean tail top-follower share",
     "tail_top_follower_share", "Experiment A: tail top-follower share"),
]


class ExpAFigures(SweepPlugin):
    name = "exp_a_figures"

    def figures(self, ctx: SweepContext) -> None:
        for y_field, ylabel, stem, title in METRIC_FIGURES:
            plot_metric(
                ctx.aggregates,
                x_field="num_states",
                y_field=y_field,
                ylabel=ylabel,
                output_file=ctx.output_dir / f"pu_scaling_{stem}_{ctx.mode}.png",
                line_by=("mode", "reward_model"),
                title=title,
            )

        progression = ctx.side_tables.get("progression", [])
        if not progression:
            return

        seen = {(r["reward_model"], r["num_states"]) for r in progression}
        for reward_model, num_states in sorted(seen):
            rows = [
                r for r in progression
                if r["reward_model"] == reward_model
                and int(r["num_states"]) == int(num_states)
                and r["mode"] == ctx.mode
            ]
            plot_series_by_group(
                rows,
                x_field="t",
                y_field="top_followers",
                group_field="seed",
                xlabel="timestep",
                ylabel="top followers",
                title=(f"Experiment A progression | {ctx.mode} | "
                       f"{reward_model} | states={num_states}"),
                output_file=(ctx.output_dir /
                             f"pu_progression_{ctx.mode}_{reward_model}_S{num_states}.png"),
            )

        # Section 5.1 report figure: max follower count stays at zero.
        report_dir = ctx.output_dir.parent / "final_report_figures"
        plot_mean_band(
            progression,
            x_field="t",
            y_field="top_followers",
            xlabel="Timestep",
            ylabel="Maximum number of followers",
            title="No follower structure emerges ($\\gamma=0, \\kappa=0$)",
            output_file=report_dir / "expA_followers_timeseries.png",
            ylim=(-0.5, 1),
            hline=0.0,
        )
        print(f"Wrote report figure: {report_dir / 'expA_followers_timeseries.png'}")


# ----------------------------------------------------------- experiment ---

EXPERIMENT = Experiment(
    name="pu_scaling",
    description="Experiment A: personal-utility baseline (gamma = 0, kappa = 0)",
    build_parser=build_parser,
    build_axes=build_axes,
    run_plugins=(
        MetricsPlugin("core", CORE_COLUMNS, measure_core),
        FollowerProgressionPlugin(table="progression"),
        AgentTracePlugin(table="agent_traces"),
    ),
    sweep_plugins=(ExpAFigures(),),
    record_columns=(
        "mode", "reward_model", "num_states", "seed", *CORE_COLUMNS,
    ),
    aggregate_spec=(
        Triple("final_top_followers"),
        Triple("time_to_50pct_followers", where=nonneg),
        Triple("time_to_90pct_followers", where=nonneg),
        Triple("leader_switches"),
        Triple("tail_welfare"),
        Triple("final_pu"),
        Triple("final_rep"),
        Triple("final_status"),
        Triple("tail_top_follower_share"),
    ),
    # gamma = kappa = 0 is what makes this the control; it is not swept and not
    # a CLI flag, so it is stated once here.
    config_overrides={"gamma": 0.0, "kappa": 0.0},
    seed_comparison_columns=(
        "mode", "reward_model", "num_states", "seed", "leader_id",
        "leader_role_final", "final_top_followers", "time_to_50pct_followers",
        "time_to_90pct_followers", "leader_switches", "tail_welfare",
        "final_pu", "final_rep", "final_status", "tail_top_follower_share",
    ),
    seed_comparison_sort=("reward_model", "num_states", "seed", "mode"),
    side_table_files={
        "progression": "pu_progression",
        "agent_traces": "pu_agent_traces",
    },
    progress_line=lambda r: (
        f"mode={r['mode']} reward={r['reward_model']} states={r['num_states']} "
        f"seed={r['seed']} leader={r['leader_id']} "
        f"top_followers={r['final_top_followers']} role={r['leader_role_final']} "
        f"final_pu={r['final_pu']} final_rep={r['final_rep']} "
        f"final_status={r['final_status']} tail_welfare={r['tail_welfare']:.4f}"
    ),
)


if __name__ == "__main__":
    EXPERIMENT.main()
