"""
Perturbation recovery as a PLUGIN on the status-scaling grid.

This is the payoff of the plugin design over a run_protocol override: the run
below sweeps gamma x kappa x perturb_strength as ONE grid. An override could
not contribute the strength axis.
"""
from harness.base import run_cli
from harness.experiments import StatusScaling
from harness.plugins import ConvergenceStop, LeaderNormPerturbation


class PerturbedStatusScaling(StatusScaling):
    name = "perturbation_recovery"
    plugins = (ConvergenceStop(), LeaderNormPerturbation())


if __name__ == "__main__":
    run_cli(PerturbedStatusScaling())
