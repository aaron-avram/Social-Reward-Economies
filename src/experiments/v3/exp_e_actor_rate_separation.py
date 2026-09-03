"""
Experiment E: separation in actor interaction rates.

Sweeps gamma x kappa and asks two questions:

  1. When an opinion leader emerges, does its actor interaction rate separate
     from everyone else's?
  2. When no leader emerges, is there any separation at all -- a top group that
     is not a leader, or none?

Both are answered by the same measurements, because the separation statistic
(`sep_gap_excess`, from the largest gap in the sorted per-agent rates) never
assumes a leader exists. The leader-conditional columns sit alongside it, so
leaderless runs are visible rather than silently dropped.

READ harness/extras/interaction_rates.py BEFORE INTERPRETING RESULTS. Two things
in particular:

  * Actor rates are clipped to [0, M] and Eq. (13)'s fixed point is
    mu* = (M + ln(H/u_0))/2, so at the defaults the whole observable range of mu
    covers only H in [0.037, 0.272]. Outside that band every agent pins to the
    same boundary and measured separation is zero regardless of the true driver
    gap. Always check `mean_share_at_ceiling` first -- if it is near 1.0, the
    run says nothing about separation, and you should raise --u-0 (or --M) and
    rerun. The `H_*` columns record the uncensored driver for exactly this
    reason.

  * Separation in rates is downstream of which term wins
    H_i = max{J^pu, gamma*J^r, kappa*J^s}. `driver_share_status` tells you
    whether the status term is winning for anyone at all. If it is zero, no
    amount of follower structure will separate the rates, because kappa*J^s
    never enters the max.

WHY THE THIRD AXIS
------------------
gamma and kappa alone under-determine the answer. --initial-actor-rate is swept
as an optional third axis because Eq. (13) is a gradient flow toward mu*: if all
agents start at the same rate and that rate is already at the ceiling, no
separation can ever appear, and a null result would be an artefact of the
initial condition rather than a property of the dynamics. Sweeping it
distinguishes the two.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.harness.aggregate import Derived, Triple, finite, nonneg  # noqa: E402
from experiments.harness.axes import Axis  # noqa: E402
from experiments.harness.cli import add_core_arguments, parse_csv_floats  # noqa: E402
from experiments.harness.experiment import Experiment  # noqa: E402
from experiments.harness.extras.interaction_rates import (  # noqa: E402
    RECORD_COLUMNS,
    ActorRateSeparationPlugin,
)
from experiments.harness.plotting import plot_errorbar, plot_heatmap  # noqa: E402
from experiments.harness.plugins import SweepContext, SweepPlugin  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Actor interaction rate separation over a gamma x kappa grid."
    )
    add_core_arguments(
        parser,
        defaults={
            "num_agents": 50,
            "num_steps": 10000,
            "seeds": 10,
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
    g = parser.add_argument_group("experiment E sweep")
    g.add_argument("--gammas", type=str, default="0,1,2,3,4")
    g.add_argument("--kappas", type=str, default="0,0.01,0.02,0.05,0.1")
    g.add_argument(
        "--initial-actor-rates", type=str, default="",
        help="Optional third axis over the common initial actor rate. Empty "
             "uses --initial-actor-rate as a fixed value. See the module "
             "docstring for why this matters.",
    )
    g.add_argument("--num-states", type=int, default=3)
    g.add_argument(
        "--reward-model",
        choices=[
            "simple_preferred_action",
            "shared_base_gaussian",
            "shared_good_bad_heterogeneous",
            "consensus_welfare_gaussian",
        ],
        default="simple_preferred_action",
    )
    return parser


def build_axes(args: argparse.Namespace) -> Sequence[Axis]:
    axes = [
        Axis.of("gamma", parse_csv_floats(args.gammas)),
        Axis.of("kappa", parse_csv_floats(args.kappas)),
    ]
    rates = parse_csv_floats(args.initial_actor_rates)
    if rates:
        # The engine field name, so the grid cell flows straight into make_config.
        axes.append(Axis.of("initial_actor_rate", rates))
    return tuple(axes)


# ------------------------------------------------------- group-level columns ---

def _mean_where(field: str, predicate):
    """Mean of `field` over the subset of a group satisfying `predicate`."""
    def fn(group) -> float:
        vals = [float(r[field]) for r in group
                if predicate(r) and np.isfinite(float(r[field]))]
        return float(np.mean(vals)) if vals else float("nan")
    return fn


def _rate(predicate):
    def fn(group) -> float:
        return float(np.mean([1.0 if predicate(r) else 0.0 for r in group]))
    return fn


AGGREGATE_SPEC = (
    # Interpretation guards come first. Every mean below is meaningless if
    # ceiling_rate is near 1, and separation cannot exist if status never wins.
    Derived("leader_rate_of_runs", _rate(lambda r: int(r["has_leader"]) == 1)),
    Derived("n_with_leader", lambda g: int(sum(int(r["has_leader"]) == 1 for r in g))),
    Derived("n_without_leader", lambda g: int(sum(int(r["has_leader"]) == 0 for r in g))),

    Triple("share_at_ceiling"),
    Triple("share_at_floor"),
    Triple("driver_share_pu"),
    Triple("driver_share_rep"),
    Triple("driver_share_status"),

    # Leader-free separation, over all runs.
    Triple("sep_gap"),
    Triple("sep_gap_excess"),
    Triple("sep_top_group_share"),
    Triple("top1_minus_rest_mean"),
    Triple("top1_z", finite_only=True),
    Triple("rate_gini"),
    Triple("rate_std"),

    # The same separation split by whether a leader emerged. This pair is the
    # direct answer to "is there separation when there is no leader" -- comparing
    # them within a cell controls for gamma and kappa.
    Derived("mean_sep_gap_excess_with_leader",
            _mean_where("sep_gap_excess", lambda r: int(r["has_leader"]) == 1)),
    Derived("mean_sep_gap_excess_no_leader",
            _mean_where("sep_gap_excess", lambda r: int(r["has_leader"]) == 0)),
    Derived("mean_rate_std_with_leader",
            _mean_where("rate_std", lambda r: int(r["has_leader"]) == 1)),
    Derived("mean_rate_std_no_leader",
            _mean_where("rate_std", lambda r: int(r["has_leader"]) == 0)),

    # Leader-conditional: NaN-dropping, so cells with no leader contribute
    # nothing rather than dragging the mean toward a sentinel.
    Triple("leader_minus_nonleader", finite_only=True),
    Triple("leader_z", finite_only=True),
    Triple("leader_rank", where=nonneg),
    Triple("leader_is_top1", where=nonneg),
    Triple("leader_in_top_group", where=nonneg),

    # The uncensored driver: separation that clipping may have hidden.
    Triple("H_leader_minus_nonleader", finite_only=True),
    Triple("H_mean", finite_only=True),
    Triple("H_max", finite_only=True),

    Triple("spearman_rate_followers", finite_only=True),
    Triple("spearman_rate_H", finite_only=True),
)


# -------------------------------------------------------------------- figures ---

class ActorRateFigures(SweepPlugin):
    name = "actor_rate_figures"

    HEATMAPS = [
        ("mean_leader_minus_nonleader", "Leader rate minus non-leader mean",
         "expE_heatmap_leader_gap.png"),
        ("mean_leader_z", "Leader gap in non-leader SDs", "expE_heatmap_leader_z.png"),
        ("mean_sep_gap_excess", "Largest-gap excess over even-spread null",
         "expE_heatmap_sep_gap_excess.png"),
        ("mean_sep_top_group_share", "Share of agents in the top rate group",
         "expE_heatmap_top_group_share.png"),
        ("mean_driver_share_status", "Share of agents driven by kappa*J^s",
         "expE_heatmap_driver_status.png"),
        ("mean_share_at_ceiling", "Share of agents at the rate ceiling",
         "expE_heatmap_ceiling.png"),
        ("mean_rate_gini", "Gini of actor rates", "expE_heatmap_rate_gini.png"),
        ("mean_H_leader_minus_nonleader", "Uncensored driver gap H",
         "expE_heatmap_H_gap.png"),
        ("leader_rate_of_runs", "Fraction of runs with an opinion leader",
         "expE_heatmap_leader_emergence.png"),
        ("mean_spearman_rate_followers", "Spearman(actor rate, followers)",
         "expE_heatmap_spearman.png"),
    ]

    LINES = [
        ("mean_leader_minus_nonleader", "ci95_leader_minus_nonleader",
         "Leader rate minus non-leader mean", "Leader actor-rate gap vs $\\kappa$",
         "expE_leader_gap_vs_kappa.png"),
        ("mean_sep_gap_excess", "ci95_sep_gap_excess",
         "Largest-gap excess (1 = even spread)", "Rate separation vs $\\kappa$",
         "expE_sep_gap_excess_vs_kappa.png"),
        ("mean_driver_share_status", "ci95_driver_share_status",
         "Share driven by $\\kappa J^s$", "Status-driven agents vs $\\kappa$",
         "expE_driver_status_vs_kappa.png"),
        ("mean_share_at_ceiling", "ci95_share_at_ceiling",
         "Share at the rate ceiling", "Ceiling saturation vs $\\kappa$",
         "expE_ceiling_vs_kappa.png"),
    ]

    def figures(self, ctx: SweepContext) -> None:
        fig_dir = ctx.output_dir / "figures"
        gammas = {r["gamma"] for r in ctx.aggregates}
        kappas = {r["kappa"] for r in ctx.aggregates}

        if len(gammas) > 1 and len(kappas) > 1:
            for value_field, title, filename in self.HEATMAPS:
                plot_heatmap(
                    ctx.aggregates, row_field="gamma", col_field="kappa",
                    value_field=value_field, title=title,
                    output_file=fig_dir / filename,
                )

        for mean_f, ci_f, ylabel, title, filename in self.LINES:
            plot_errorbar(
                ctx.aggregates, x_field="kappa", mean_field=mean_f, err_field=ci_f,
                ylabel=ylabel, title=title, xlabel="$\\kappa$",
                output_file=fig_dir / filename, line_by=("gamma",),
            )

        self._plot_leader_vs_leaderless(ctx, fig_dir)
        self._plot_rate_distributions(ctx, fig_dir)
        self._plot_timeseries(ctx, fig_dir)
        print(f"[\u2713] Experiment E figures saved to {fig_dir}")

    # -- the figure that answers the actual question -----------------------

    def _plot_leader_vs_leaderless(self, ctx: SweepContext, fig_dir: Path) -> None:
        """Separation with a leader vs without, in the same cell.

        Plotting both series against kappa on shared axes is the point: it holds
        gamma and kappa fixed while varying only whether a leader emerged, so a
        difference cannot be attributed to the sweep parameters.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [r for r in ctx.aggregates]
        if not rows:
            return

        gammas = sorted({r["gamma"] for r in rows})
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(1, len(gammas) - 1)) for i in range(len(gammas))]

        plt.figure(figsize=(7.4, 4.8))
        for color, gamma in zip(colors, gammas):
            series = sorted([r for r in rows if r["gamma"] == gamma],
                            key=lambda r: r["kappa"])
            xs = [r["kappa"] for r in series]
            with_leader = [r["mean_sep_gap_excess_with_leader"] for r in series]
            no_leader = [r["mean_sep_gap_excess_no_leader"] for r in series]
            plt.plot(xs, with_leader, "-o", color=color, linewidth=1.8,
                     label=f"$\\gamma$={gamma:g}, leader")
            plt.plot(xs, no_leader, "--s", color=color, linewidth=1.4,
                     alpha=0.75, label=f"$\\gamma$={gamma:g}, no leader")

        plt.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        plt.text(
            plt.xlim()[1], 1.0, " even spread ", va="center", ha="right", fontsize=8,
        )
        plt.xlabel("$\\kappa$")
        plt.ylabel("Largest-gap excess over even-spread null")
        plt.title("Rate separation: runs with a leader vs runs without")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_dir / "expE_separation_leader_vs_leaderless.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    def _plot_rate_distributions(self, ctx: SweepContext, fig_dir: Path) -> None:
        """Per-agent rate strip plots, leader highlighted, one panel per cell.

        The aggregate statistics compress a distribution to a number; this shows
        the distribution, which is where a bimodal split is actually visible.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = ctx.side_tables.get("agent_rates", [])
        if not rows:
            return

        gammas = sorted({r["gamma"] for r in rows})
        kappas = sorted({r["kappa"] for r in rows})
        if not gammas or not kappas:
            return

        fig, axes = plt.subplots(
            len(gammas), len(kappas),
            figsize=(2.7 * len(kappas), 2.3 * len(gammas)),
            squeeze=False, sharey=True,
        )
        rng = np.random.default_rng(0)

        for i, gamma in enumerate(gammas):
            for j, kappa in enumerate(kappas):
                ax = axes[i][j]
                cell = [r for r in rows
                        if r["gamma"] == gamma and r["kappa"] == kappa]
                if not cell:
                    ax.set_axis_off()
                    continue

                others = [r for r in cell if not int(r["is_leader"])]
                leaders = [r for r in cell if int(r["is_leader"])]
                for group, color, size, alpha, label in (
                    (others, "tab:blue", 6, 0.30, "non-leader"),
                    (leaders, "tab:red", 26, 0.95, "leader"),
                ):
                    if not group:
                        continue
                    ys = [float(r["actor_rate"]) for r in group]
                    xs = rng.normal(0.0, 0.09, size=len(ys))
                    ax.scatter(xs, ys, s=size, alpha=alpha, color=color,
                               label=label, edgecolors="none")

                ax.set_xlim(-0.45, 0.45)
                ax.set_xticks([])
                if i == 0:
                    ax.set_title(f"$\\kappa$={kappa:g}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"$\\gamma$={gamma:g}\nactor rate", fontsize=8)
                ax.grid(alpha=0.2, axis="y")

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
        fig.suptitle("Per-agent tail actor rates by cell (all seeds pooled)",
                     fontsize=12)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(fig_dir / "expE_rate_distributions.png", dpi=190)
        plt.close(fig)

    def _plot_timeseries(self, ctx: SweepContext, fig_dir: Path) -> None:
        """Leader vs non-leader rate over time, with a p10-p90 band."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = ctx.side_tables.get("rate_timeseries", [])
        if not rows:
            return

        cells = sorted({(r["gamma"], r["kappa"], r["seed"]) for r in rows})
        for gamma, kappa, seed in cells:
            series = sorted(
                [r for r in rows
                 if r["gamma"] == gamma and r["kappa"] == kappa and r["seed"] == seed],
                key=lambda r: r["t"],
            )
            if not series:
                continue

            t = [r["t"] for r in series]
            plt.figure(figsize=(7.2, 4.4))
            plt.fill_between(t, [r["p10_rate"] for r in series],
                             [r["p90_rate"] for r in series],
                             alpha=0.20, color="tab:blue", label="p10-p90")
            plt.plot(t, [r["nonleader_mean_rate"] for r in series],
                     color="tab:blue", linewidth=1.7, label="non-leader mean")
            if any(np.isfinite(float(r["leader_rate"])) for r in series):
                plt.plot(t, [r["leader_rate"] for r in series],
                         color="tab:red", linewidth=2.0, label="leader")
            plt.xlabel("timestep")
            plt.ylabel("actor interaction rate")
            plt.title(f"Actor rates | $\\gamma$={gamma:g} $\\kappa$={kappa:g} "
                      f"seed={seed}")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=8)
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                fig_dir / f"expE_rate_timeseries_g{gamma:g}_k{kappa:g}_seed{seed}.png",
                dpi=180,
            )
            plt.close()


# ----------------------------------------------------------------- experiment ---

def _axis_names(args) -> List[str]:
    names = ["gamma", "kappa"]
    if parse_csv_floats(args.initial_actor_rates):
        names.append("initial_actor_rate")
    return names


class _ExperimentE(Experiment):
    """Adjusts the record schema when the optional third axis is present.

    Key columns are part of the CSV contract, so adding an axis has to add a
    column. Overriding `validate` keeps the column-ownership check honest
    instead of loosening it.
    """

    def validate(self, axis_names: Sequence[str]) -> None:
        object.__setattr__(
            self, "record_columns",
            ("mode", *axis_names, "seed", *RECORD_COLUMNS),
        )
        super().validate(axis_names)


EXPERIMENT = _ExperimentE(
    name="actor_rate_separation",
    description=(
        "Experiment E: separation of actor interaction rates over a gamma x kappa grid"
    ),
    build_parser=build_parser,
    build_axes=build_axes,
    run_plugins=(ActorRateSeparationPlugin(),),
    sweep_plugins=(ActorRateFigures(),),
    record_columns=("mode", "gamma", "kappa", "seed", *RECORD_COLUMNS),
    aggregate_spec=AGGREGATE_SPEC,
    side_table_files={
        "agent_rates": "actor_rate_agents",
        "rate_timeseries": "actor_rate_timeseries",
    },
    seed_comparison_columns=(
        "mode", "gamma", "kappa", "seed", "has_leader", "leader_id",
        "leader_followers", "leader_rate", "nonleader_rate_mean",
        "leader_minus_nonleader", "leader_z", "leader_rank",
        "sep_gap_excess", "sep_top_group_size", "rate_gini",
        "share_at_ceiling", "driver_share_status", "leader_driver",
    ),
    seed_comparison_sort=("mode", "gamma", "kappa", "seed"),
    progress_line=lambda r: (
        f"gamma={r['gamma']:g} kappa={r['kappa']:g} seed={r['seed']} "
        f"leader={r['leader_id']} "
        f"leader_gap={r['leader_minus_nonleader']:.4f} "
        f"leader_z={r['leader_z']:.2f} "
        f"sep_excess={r['sep_gap_excess']:.2f} "
        f"top_group={r['sep_top_group_size']} "
        f"ceiling={r['share_at_ceiling']:.2f} "
        f"status_driven={r['driver_share_status']:.2f}"
    ),
)


if __name__ == "__main__":
    EXPERIMENT.main()
