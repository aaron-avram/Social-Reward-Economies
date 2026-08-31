"""The four original experiments, expressed on the base harness."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np

from model.config import RewardModelKind
from harness.base import (
    Axis, Experiment, parse_floats, parse_ints, parse_strings,
    with_algorithm, with_dims, with_reward,
)
from harness.plugin import RunContext


def _common_metrics(ctx: RunContext) -> dict[str, Any]:
    """Outcome columns every scaling experiment reports."""
    counts = ctx.follower_counts()
    roles = ctx.role_counts()
    welfare = ctx.results.paper_welfare_followers_only
    tail = int(getattr(ctx.args, "tail_window", 200))
    return {
        "leader": ctx.opinion_leader(),
        "top_followers": max(counts, default=0),
        "final_pu": roles["personal_utility"],
        "final_rep": roles["reputation"],
        "final_status": roles["status"],
        "n_role_updates": len(ctx.results.role_update_times),
        "tail_welfare": float(np.mean(welfare[-tail:])) if welfare else float("nan"),
        "final_welfare": float(welfare[-1]) if welfare else float("nan"),
    }


class StatusScaling(Experiment):
    """gamma x kappa sweep. Was status_scaling.py."""

    name = "status_scaling"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--gammas", type=str, default="0,1,2,3,4")
        parser.add_argument("--kappas", type=str, default="0,0.01,0.02,0.05,0.1")

    def axes(self, args: Namespace) -> list[Axis]:
        return [
            Axis("gamma", parse_floats(args.gammas),
                 apply=lambda cfg, v: with_algorithm(cfg, gamma=v)),
            Axis("kappa", parse_floats(args.kappas),
                 apply=lambda cfg, v: with_algorithm(cfg, kappa=v)),
        ]

    def measure(self, ctx: RunContext) -> dict[str, Any]:
        return _common_metrics(ctx)


class ReputationStatusScaling(StatusScaling):
    """Same axes, different defaults. Was reputation_status_scaling.py."""

    name = "reputation_status_scaling"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--gammas", type=str, default="0,2,4,6,8")
        parser.add_argument("--kappas", type=str, default="0,1,2,4,8")


class PuScaling(Experiment):
    """
    reward_model x num_states sweep. Was pu_scaling.py.

    Shows that axes need not be numeric: one varies an enum, the other a
    dimension.
    """

    name = "pu_scaling"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--reward-models", type=str,
                            default="shared_base_gaussian")
        parser.add_argument("--num-states-list", type=str, default="2,3,4")

    def axes(self, args: Namespace) -> list[Axis]:
        return [
            Axis("reward_model_swept", parse_strings(args.reward_models),
                 apply=lambda cfg, v: with_reward(cfg, kind=RewardModelKind(v))),
            Axis("num_states_swept", parse_ints(args.num_states_list),
                 apply=lambda cfg, v: with_dims(cfg, num_states=v)),
        ]

    def measure(self, ctx: RunContext) -> dict[str, Any]:
        return _common_metrics(ctx)
