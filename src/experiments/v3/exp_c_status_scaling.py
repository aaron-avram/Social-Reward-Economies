"""
Experiment C: status scaling.

Sweeps gamma x kappa and asks whether raising the status weight makes the
emergent opinion leader a STATUS agent, and what that does to welfare and to
the optimality of the norm the leader ends up broadcasting.

Ported from src/experiments/v2/status_scaling.py (1117 lines). CSV output is
byte-identical for the same arguments; see tools/parity_check.py.

BEHAVIOUR PRESERVED DELIBERATELY
--------------------------------
The v2 make_config declared --c-threshold/--B-R/--B-F on the CLI and then
hardcoded c_threshold=0.1, B_R=0.3, B_F=0.2, so those three flags did nothing.
That is reproduced here via `config_overrides` rather than silently fixed,
because fixing it would invalidate every committed Experiment C figure. The
flags are still accepted and still ignored -- but now the fact that they are
ignored is stated in one visible place instead of hidden in a 55-line function.
Pass --respect-threshold-flags to opt into the corrected behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.harness.aggregate import Triple, finite, nonneg  # noqa: E402
from experiments.harness.axes import Axis  # noqa: E402
from experiments.harness.cli import add_core_arguments, parse_csv_floats  # noqa: E402
from experiments.harness.experiment import Experiment  # noqa: E402
from experiments.harness.extras.leader_status import BASE_COLUMNS, LeaderStatusPlugin  # noqa: E402
from experiments.harness.extras.norm_optimality import NormOptimalityPlugin  # noqa: E402
from experiments.harness.plotting import plot_errorbar, plot_heatmap, plot_metric  # noqa: E402
from experiments.harness.plugins import SweepContext, SweepPlugin  # noqa: E402

NORM_COLUMNS = NormOptimalityPlugin.columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Status scaling kappa sweep harness.")
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
    group = parser.add_argument_group("experiment C sweep")
    group.add_argument("--gammas", type=str, default="0,1,2,3,4")
    group.add_argument("--kappas", type=str, default="0,0.01,0.02,0.05,0.1")
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
        "--respect-threshold-flags",
        action="store_true",
        help="Honour --c-threshold/--B-R/--B-F instead of the hardcoded "
             "0.1/0.3/0.2 the v2 harness used. Changes results; off by default "
             "so committed figures remain reproducible.",
    )
    group.add_argument("--plot-sample-interval", type=int, default=100)
    return parser


def build_axes(args: argparse.Namespace) -> Sequence[Axis]:
    return (
        Axis.of("gamma", parse_csv_floats(args.gammas)),
        Axis.of("kappa", parse_csv_floats(args.kappas)),
    )


class ExpCFigures(SweepPlugin):
    name = "exp_c_figures"

    METRIC_FIGURES = [
        ("mean_final_top_followers", "Mean final top followers", "final_top_followers"),
        ("mean_time_to_90pct_followers", "Mean time to 90% followers", "time_to_90pct"),
        ("mean_leader_switches", "Mean leader switches", "leader_switches"),
        ("mean_tail_welfare", "Mean tail welfare", "tail_welfare"),
        ("mean_leader_is_status_final", "P(final leader is STATUS)", "leader_is_status"),
        ("mean_final_status_count", "Mean final status count", "final_status_count"),
        ("mean_tail_status_leader_share", "Tail share: top leader in STATUS",
         "tail_status_leader_share"),
        ("mean_tail_status_agent_share", "Tail share: all agents in STATUS",
         "tail_status_agent_share"),
    ]

    REPORT_FIGURES = [
        ("mean_leader_is_status_final", "ci95_leader_is_status_final",
         "Final leader is STATUS vs $\\kappa$", "Probability final leader is STATUS",
         "expC_leader_is_status_vs_kappa.png", (-0.05, 1.05)),
        ("mean_tail_welfare", "ci95_tail_welfare",
         "Tail welfare vs $\\kappa$", "Mean tail welfare",
         "expC_tail_welfare_vs_kappa.png", None),
        ("mean_welfare_gap_to_best", "ci95_welfare_gap_to_best",
         "Welfare gap to best norm vs $\\kappa$", "Mean welfare gap",
         "expC_welfare_gap_vs_kappa.png", None),
        ("mean_is_final_norm_optimal", "ci95_is_final_norm_optimal",
         "Probability final norm is optimal vs $\\kappa$", "Probability optimal",
         "expC_optimal_norm_probability_vs_kappa.png", (-0.05, 1.05)),
    ]

    HEATMAPS = [
        ("mean_leader_is_status_final", "P(final leader is STATUS)",
         "expC_heatmap_leader_is_status.png"),
        ("mean_tail_welfare", "Mean tail welfare", "expC_heatmap_tail_welfare.png"),
        ("mean_welfare_gap_to_best", "Mean welfare gap to best norm",
         "expC_heatmap_welfare_gap.png"),
        ("mean_is_final_norm_optimal", "P(final norm is optimal)",
         "expC_heatmap_optimal_norm.png"),
    ]

    def figures(self, ctx: SweepContext) -> None:
        for y_field, ylabel, stem in self.METRIC_FIGURES:
            plot_metric(
                ctx.aggregates,
                x_field="kappa",
                y_field=y_field,
                ylabel=ylabel,
                output_file=ctx.output_dir / f"status_scaling_{stem}_{ctx.mode}.png",
                line_by=("mode", "gamma"),
                label_fmt="{mode} | gamma={gamma:g}",
            )

        fig_dir = ctx.output_dir.parent / "final_report_figures"
        for mean_f, ci_f, title, ylabel, filename, _ylim in self.REPORT_FIGURES:
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

        gammas = {r["gamma"] for r in ctx.aggregates}
        kappas = {r["kappa"] for r in ctx.aggregates}
        if len(gammas) > 1 and len(kappas) > 1:
            for value_field, title, filename in self.HEATMAPS:
                plot_heatmap(
                    ctx.aggregates,
                    row_field="gamma",
                    col_field="kappa",
                    value_field=value_field,
                    title=title,
                    output_file=fig_dir / filename,
                )
        print(f"[\u2713] Report figures saved to {fig_dir}")


EXPERIMENT = Experiment(
    name="status_scaling",
    description="Experiment C: status scaling over a gamma x kappa grid",
    build_parser=build_parser,
    build_axes=build_axes,
    run_plugins=(
        # margin=0 reproduces v2: switches and the tail leader share are both
        # computed off the plain lowest-id-tie-break series.
        LeaderStatusPlugin(threshold_frac=0.90, leader_switch_margin=0),
        NormOptimalityPlugin(),
    ),
    sweep_plugins=(ExpCFigures(),),
    record_columns=("mode", "gamma", "kappa", "seed", *BASE_COLUMNS, *NORM_COLUMNS),
    aggregate_spec=(
        Triple("final_top_followers"),
        Triple("time_to_90pct_followers", where=nonneg),
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
        f"optimal={r['is_final_norm_optimal']} gap={r['welfare_gap_to_best']:.6f}"
    ),
)


def _apply_legacy_thresholds(argv=None) -> None:
    """v2 pinned these three regardless of CLI. See the module docstring."""
    import sys as _sys

    if "--respect-threshold-flags" not in (argv if argv is not None else _sys.argv):
        EXPERIMENT.config_overrides.update(
            {"c_threshold": 0.1, "B_R": 0.3, "B_F": 0.2}
        )


if __name__ == "__main__":
    _apply_legacy_thresholds()
    EXPERIMENT.main()
