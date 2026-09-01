"""
Experiment D: perturbation and recovery.

Run until an opinion leader emerges, force that leader to behave badly for a
fixed window, then measure whether and how leadership recovers -- and whether
the leader who comes back is the same one.

Ported from src/experiments/v2/perturbation_recovery.py (1684 lines). The state
machine lives in harness/extras/perturbation.py; this file is the declaration.

WHY THIS IS A 1x1 GRID
----------------------
D sweeps only seeds: --gamma and --kappa are scalars. Declaring them as
single-valued axes rather than special-casing "no axes" means the CSV key
columns (mode, gamma, kappa, seed) and the aggregate grouping come out right
with no extra machinery, and widening the sweep later is a one-line change --
pass a comma-separated list and D becomes a grid like B and C.

TWO LEGACY BEHAVIOURS PRESERVED
-------------------------------
1. D's async path did not refresh the tracked state after a role update, unlike
   A/B/C. That means step t does NOT reflect the post-update follower graph, so
   D's follower series lags the others by a step. Reproduced via
   `async_refresh=False`; almost certainly an oversight rather than a choice.
2. D's local `_mean_std_ci` dropped non-finite values before averaging, while
   A/B/C filtered explicitly per call site. Reproduced via `finite_only=True`.
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

from experiments.harness.aggregate import Derived, Triple, positive  # noqa: E402
from experiments.harness.axes import Axis  # noqa: E402
from experiments.harness.cli import add_core_arguments  # noqa: E402
from experiments.harness.experiment import Experiment  # noqa: E402
from experiments.harness.extras.perturbation import (  # noqa: E402
    RECORD_COLUMNS,
    PerturbationFigures,
    PerturbationRecoveryPlugin,
    build_run_subdir_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perturbation and recovery of an established opinion leader."
    )
    add_core_arguments(
        parser,
        steps_flag="--num-steps-max",
        defaults={
            "num_agents": 8,
            "num_steps": 12000,
            "seeds": 10,
            "delta": 0.15,
            "B_R": 0.3,
            "B_F": 0.15,
            "c_threshold": 0.1,
            "initial_actor_rate": 0.7,
            "initial_participant_rate": 0.7,
            "reward_base_sigma": 0.08,
            "reward_agent_sigma": 0.1,
            "role_update_base_interval": 3000,
            "output_dir": str(Path(__file__).resolve().parent / "outputs"),
        },
    )
    parser.set_defaults(mode="static")

    g = parser.add_argument_group("experiment D parameters")
    g.add_argument("--gamma", type=float, default=2.0)
    g.add_argument("--kappa", type=float, default=2.0)
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
    g.add_argument("--reward-good-value", type=float, default=1.0)
    g.add_argument("--reward-bad-value", type=float, default=0.1)
    g.add_argument("--reward-order-gap", type=float, default=0.02)

    g = parser.add_argument_group("experiment D output")
    g.add_argument("--output-prefix", type=str, default="perturbation_recovery")
    g.add_argument(
        "--run-label", type=str, default="",
        help="Subdirectory name under --output-dir for this run.",
    )
    g.add_argument(
        "--auto-run-subdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs to a parameter-stamped subdirectory, so two runs "
             "with different settings cannot overwrite each other.",
    )
    g.add_argument("--plot-sample-interval", type=int, default=250)
    return parser


def build_axes(args: argparse.Namespace) -> Sequence[Axis]:
    # Single-valued axes: see the module docstring.
    return (
        Axis.of("gamma", [float(args.gamma)]),
        Axis.of("kappa", [float(args.kappa)]),
    )


def output_dir_for(args: argparse.Namespace) -> Path:
    root = Path(args.output_dir)
    return root / build_run_subdir_name(args) if bool(args.auto_run_subdir) else root


def _rate(predicate):
    def fn(group) -> float:
        return float(np.mean([1.0 if predicate(r) else 0.0 for r in group]))
    return fn


EXPERIMENT = Experiment(
    name="perturbation_recovery",
    description="Experiment D: perturbation and recovery of an established opinion leader",
    build_parser=build_parser,
    build_axes=build_axes,
    run_plugins=(PerturbationRecoveryPlugin(),),
    sweep_plugins=(PerturbationFigures(),),
    record_columns=("mode", "gamma", "kappa", "seed", *RECORD_COLUMNS),
    aggregate_spec=(
        # The five rates come first: every mean below is conditional on the
        # corresponding event happening, so the rates are what make them
        # interpretable. mean_recovery_time in particular is conditional on
        # recovering at all -- read it with recovery_rate, never alone.
        Derived("conv_rate", _rate(lambda r: r["converged"])),
        Derived("drop_rate", _rate(
            lambda r: np.isfinite(r["drop_fraction"]) and r["drop_fraction"] > 0)),
        Derived("normless_rate", _rate(lambda r: r["normless_duration"] > 0)),
        Derived("recovery_rate", _rate(lambda r: r["recovery_time"] > 0)),
        Derived("stable_recovery_rate", _rate(lambda r: r["stable_recovery"])),
        Triple("drop_fraction", finite_only=True),
        Triple("normless_duration", finite_only=True),
        Triple("recovery_time", where=positive, finite_only=True),
        Triple("final_top_followers", finite_only=True),
    ),
    steps_attr="num_steps_max",
    output_dir_fn=output_dir_for,
    file_stem_fn=lambda args: str(args.output_prefix),
    # See the module docstring, note 1.
    async_refresh=False,
    progress_line=lambda r: (
        f"seed={r['seed']} converged={r['converged']} t_conv={r['t_conv']} "
        f"leader_pre={r['leader_pre']} drop_frac={r['drop_fraction']:.3f} "
        f"normless={r['normless_duration']} recovery_t={r['recovery_time']} "
        f"leader_changed={r['leader_changed']} stable={r['stable_recovery']}"
    ),
)


if __name__ == "__main__":
    EXPERIMENT.main()
