"""
Leader-norm perturbation, as PLUGINS on the scaling grid.

Unlike v1/v2, perturbation strength is a grid axis, so this sweeps
gamma x kappa x perturb_strength in a single run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # sandbox layout; use parents[3] under src/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.base import run_cli
from harness.experiments import StatusScaling
from harness.plugins import ConvergenceStop, LeaderNormPerturbation


class PerturbationRecovery(StatusScaling):
    name = "perturbation_recovery"
    plugins = (ConvergenceStop(), LeaderNormPerturbation())


if __name__ == "__main__":
    run_cli(PerturbationRecovery())
