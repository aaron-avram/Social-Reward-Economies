"""
Experiment B: reputation x status scaling.

The full gamma x kappa grid: how status weight affects the stability and the
speed of convergence to an opinion leader, at each level of reputation weight.

Ported from src/experiments/v2/reputation_status_scaling.py (1435 lines). CSV
output is byte-identical for the same arguments; see tools/parity_check.py.

Three corrections that were made in v2 and are preserved here, because each one
changes results and each is easy to lose in a refactor:

  * --c-threshold/--B-R/--B-F are actually honoured (status_scaling declared
    them and ignored them).
  * --kappa-scale-by-n divides the swept kappa-tilde by N before it reaches the
    engine, so the status term does not swamp reputation at realistic N.
  * --leader-switch-margin defaults to 1. The plain leader series breaks ties by
    lowest agent id, and kappa makes near-ties more common, so counting switches
    off the plain series conflates instability with tied leadership -- biasing
    exactly what this sweep measures.

READING THE OUTPUT
------------------
mean_time_to_90pct_followers is CONDITIONAL ON REACHING the threshold, and
whether a run reaches is itself a function of gamma and kappa. Always read it
alongside reach_rate; the mean alone will suggest high-kappa cells converge
faster when in fact most of them never converge at all.
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

from experiments.harness.aggregate import Derived, Triple, finite, nonneg, positive  # noqa: E402
from experiments.harness.axes import Axis  # noqa: E402
from experiments.harness.cli import add_core_arguments, parse_csv_floats  # noqa: E402
from experiments.harness.experiment import Experiment  # noqa: E402
from experiments.harness.extras.leader_status import (  # noqa: E402
    BASE_COLUMNS,
    CENSORING_COLUMNS,
    LeaderStatusPlugin,
)
from experiments.harness.extras.norm_optimality import NormOptimalityPlugin  # noqa: E402
from experiments.harness.extras.tracking import (  # noqa: E402
    ActorRateTrackerPlugin,
    ConsensusTrackerPlugin,
)
from experiments.harness.plotting import plot_errorbar, plot_heatmap  # noqa: E402
from experiments.harness.plugins import RunContext, RunPlugin, SweepContext, SweepPlugin  # noqa: E402

NORM_COLUMNS = NormOptimalityPlugin.columns
CONSENSUS_COLUMNS = ConsensusTrackerPlugin.columns
ACTOR_RATE_COLUMNS = ActorRateTrackerPlugin.columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reputation x status scaling grid harness."
    )
    add_core_arguments(
        parser,
        defaults={
            "num_agents": 100,
            "num_steps": 10000,
            "seeds": 20,
            "delta": 0.15,
            "initial_actor_rate": 0.2,
            "initial_participant_rate": 0.2,
            "reward_base_sigma": 0.08,
            "reward_agent_sigma": 0.1,
            "c_threshold": 0.1,
            "B_R": 0.8,
            "B_F": 0.6,
            "role_update_base_interval": 3000,
            "output_dir": str(Path(__file__).resolve().parent / "outputs"),
        },
    )
    group = parser.add_argument_group("experiment B sweep")
    group.add_argument(
        "--gammas",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated gamma values. Swept as a FULL GRID against --kappas.",
    )
    group.add_argument("--kappas", type=str, default="0,0.01,0.02,0.05,0.1")
    group.add_argument(
        "--kappa-scale-by-n",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret --kappas as kappa-tilde and pass kappa = kappa_tilde / num_agents "
             "to the engine.",
    )
    group.add_argument("--num-states", type=int, default=3)
    group.add_argument(
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
    group.add_argument(
        "--convergence-threshold-frac",
        type=float,
        default=0.90,
        help="Follower fraction (of N-1) defining convergence to an opinion leader.",
    )
    group.add_argument(
        "--leader-switch-margin",
        type=int,
        default=1,
        help="Followers by which a challenger must STRICTLY EXCEED the incumbent "
             "before a leader switch is counted. 0 reproduces lowest-agent-id "
             "tie-breaking, which inflates leader_switches under near-ties.",
    )
    group.add_argument("--plot-sample-interval", type=int, default=100)
    return parser


def build_axes(args: argparse.Namespace) -> Sequence[Axis]:
    return (
        Axis.of("gamma", parse_csv_floats(args.gammas)),
        Axis.of("kappa", parse_csv_floats(args.kappas)),
    )


class KappaScalingPlugin(RunPlugin):
    """Maps the swept kappa-tilde onto the kappa handed to the engine.

    The CSV keeps the swept value, so figures stay labelled in kappa-tilde while
    the engine sees kappa-tilde / N. A `configure` hook is exactly the right
    place for this: the grid cell is the experiment's parameter, the engine
    field is an implementation detail of that parameterisation.
    """

    name = "kappa_scaling"
    columns = ()

    def configure(self, config, ctx: RunContext):
        if not bool(getattr(ctx.args, "kappa_scale_by_n", False)):
            return config
        from dataclasses import replace

        scaled = float(ctx.cell["kappa"]) / float(ctx.args.num_agents)
        return replace(
            config, algorithm=replace(config.algorithm, kappa=scaled)
        )


def _reach_rate(group) -> float:
    reached = [r for r in group if r["time_to_90pct_followers"] >= 0]
    return float(len(reached) / len(group)) if group else float("nan")


def _n_reached(group) -> int:
    return int(sum(1 for r in group if r["time_to_90pct_followers"] >= 0))


class ExpBFigures(SweepPlugin):
    name = "exp_b_figures"

    HEATMAPS = [
        ("mean_tail_welfare", "Mean tail welfare",
         "reputation_status_scaling_tail_welfare_{mode}.png"),
        ("mean_consensus_step_first", "First consensus step",
         "mean_first_consensus_{mode}.png"),
        ("mean_consensus_step_final", "Last consensus step",
         "mean_final_consensus_{mode}.png"),
        ("mean_num_consensus", "Number of consensus episodes",
         "mean_num_consensus_{mode}.png"),
        ("reach_rate", "Fraction of runs reaching the follower threshold",
         "reach_rate_{mode}.png"),
        ("mean_leader_is_status_final", "P(final leader is STATUS)",
         "leader_is_status_{mode}.png"),
    ]

    REPORT_FIGURES = [
        ("mean_leader_is_status_final", "ci95_leader_is_status_final",
         "Final leader is STATUS vs $\\kappa$", "Probability final leader is STATUS",
         "expC_leader_is_status_vs_kappa.png"),
        ("mean_tail_welfare", "ci95_tail_welfare",
         "Tail welfare vs $\\kappa$", "Mean tail welfare",
         "expC_tail_welfare_vs_kappa.png"),
        ("mean_welfare_gap_to_best", "ci95_welfare_gap_to_best",
         "Welfare gap to best norm vs $\\kappa$", "Mean welfare gap",
         "expC_welfare_gap_vs_kappa.png"),
        ("mean_is_final_norm_optimal", "ci95_is_final_norm_optimal",
         "Probability final norm is optimal vs $\\kappa$", "Probability optimal",
         "expC_optimal_norm_probability_vs_kappa.png"),
        ("mean_tail_top_follower_share", "ci95_tail_top_follower_share",
         "Tail top-follower share vs $\\kappa$", "Mean tail top-follower share",
         "expC_tail_top_follower_share_vs_kappa.png"),
    ]

    def figures(self, ctx: SweepContext) -> None:
        for value_field, title, pattern in self.HEATMAPS:
            plot_heatmap(
                ctx.aggregates,
                row_field="gamma",
                col_field="kappa",
                value_field=value_field,
                title=title,
                output_file=ctx.output_dir / pattern.format(mode=ctx.mode),
            )

        fig_dir = ctx.output_dir.parent / "final_report_figures"
        for mean_f, ci_f, title, ylabel, filename in self.REPORT_FIGURES:
            plot_errorbar(
                ctx.aggregates,
                x_field="kappa",
                mean_field=mean_f,
                err_field=ci_f,
                ylabel=ylabel,
                title=title,
                xlabel="$\\kappa$",
                output_file=fig_dir / filename,
                line_by=("gamma",),
            )

        # reach_rate is plotted next to the conditional mean deliberately: the
        # mean is uninterpretable without it.
        plot_errorbar(
            ctx.aggregates,
            x_field="kappa",
            mean_field="reach_rate",
            err_field=None,
            ylabel="Fraction of runs reaching threshold",
            title="Reach rate vs $\\kappa$",
            xlabel="$\\kappa$",
            output_file=fig_dir / "expC_heatmap_reach_rate.png",
            line_by=("gamma",),
        )
        print(f"[\u2713] Report figures saved to {fig_dir}")


EXPERIMENT = Experiment(
    name="reputation_status_scaling",
    description="Experiment B: reputation x status scaling over the full gamma x kappa grid",
    build_parser=build_parser,
    build_axes=build_axes,
    run_plugins=(
        KappaScalingPlugin(),
        LeaderStatusPlugin(
            threshold_frac=0.90,
            leader_switch_margin=1,
            include_censoring=True,
            threshold_frac_arg="convergence_threshold_frac",
            margin_arg="leader_switch_margin",
        ),
        NormOptimalityPlugin(),
        ConsensusTrackerPlugin(threshold_frac=0.50),
        ActorRateTrackerPlugin(),
    ),
    sweep_plugins=(ExpBFigures(),),
    record_columns=(
        "mode", "gamma", "kappa", "seed",
        *BASE_COLUMNS, *NORM_COLUMNS, *CONSENSUS_COLUMNS, *ACTOR_RATE_COLUMNS,
        *CENSORING_COLUMNS,
    ),
    aggregate_spec=(
        Derived("reach_rate", _reach_rate),
        Derived("n_reached", _n_reached),
        Triple("final_top_followers"),
        # Conditional on reaching; read with reach_rate. The [-1.0] fallback
        # preserves v2's sentinel for cells where no run ever converged.
        Triple("time_to_90pct_followers", where=nonneg, fallback=[-1.0]),
        Triple("leader_switches"),
        Triple("tail_welfare"),
        Triple("leader_is_status_final"),
        Triple("final_status_count"),
        Triple("tail_status_leader_share"),
        Triple("tail_status_agent_share"),
        Triple("final_norm_welfare_check", where=finite),
        Triple("best_norm_welfare", where=finite),
        Triple("welfare_gap_to_best", where=finite),
        Triple("is_final_norm_optimal", where=nonneg),
        Triple("tail_top_follower_share"),
        Triple("consensus_step_first", where=positive),
        Triple("consensus_step_final", where=positive),
        Triple("num_consensus", where=positive),
    ),
    seed_comparison_columns=(
        "mode", "gamma", "seed", "kappa", "leader_id", "leader_role_final",
        "final_top_followers", "tail_welfare", "final_norm",
        "final_norm_welfare_check", "best_norm", "best_norm_welfare",
        "welfare_gap_to_best", "is_final_norm_optimal",
    ),
    seed_comparison_sort=("mode", "gamma", "seed", "kappa"),
    progress_line=lambda r: (
        f"mode={r['mode']} gamma={r['gamma']:g} kappa={r['kappa']:g} seed={r['seed']} "
        f"leader={r['leader_id']} top_followers={r['final_top_followers']} "
        f"leader_role={r['leader_role_final']} n_status={r['final_status_count']} "
        f"tail_welfare={r['tail_welfare']:.4f} "
        f"optimal={r['is_final_norm_optimal']} gap={r['welfare_gap_to_best']:.6f} "
        f"first_consensus={r['consensus_step_first']} "
        f"last_consensus={r['consensus_step_final']} "
        f"num_consensus={r['num_consensus']}"
    ),
)


if __name__ == "__main__":
    EXPERIMENT.main()
