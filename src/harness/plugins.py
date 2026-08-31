"""Concrete plugins. Each is a worked example of one part of the interface."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Optional

import numpy as np

from harness.base import parse_floats
from harness.plugin import Axis, Plugin, RunContext


class LeaderNormPerturbation(Plugin):
    """
    Shock the opinion leader's policy partway through the run, then measure
    whether the population recovers.

    Demonstrates all four parts of the interface: a CLI argument, a sweep axis
    (so strength is a grid dimension, not a separate experiment), a step hook
    that intervenes, and measured columns.
    """

    name = "leader_perturbation"

    def __init__(self) -> None:
        self.strength: float = 1.0
        self.reset()

    def reset(self) -> None:
        self.fired = False
        self.leader_pre = -1
        self.leader_post = -1
        self.followers_pre = 0
        self.min_followers_after = -1
        self.recovery_step: Optional[int] = None

    def add_arguments(self, parser: ArgumentParser) -> None:
        g = parser.add_argument_group("leader perturbation")
        g.add_argument("--perturb-at", type=float, default=0.5,
                       help="Fraction of the run at which to perturb.")
        g.add_argument("--perturb-strengths", type=str, default="2.0",
                       help="Comma-separated logit shifts. Swept as a grid axis.")
        g.add_argument("--recovery-frac", type=float, default=0.9,
                       help="Follower share of the pre-shock level counted as "
                            "recovered.")

    def axes(self, args: Namespace) -> list[Axis]:
        # store= rather than apply=: strength is plugin state, not a config field.
        return [Axis("perturb_strength", parse_floats(args.perturb_strengths),
                     store="strength")]

    def on_run_start(self, ctx: RunContext) -> None:
        self.fire_at = max(1, int(ctx.args.perturb_at * ctx.args.num_steps))

    def on_step(self, ctx: RunContext) -> None:
        if not self.fired and ctx.t >= self.fire_at:
            self._perturb(ctx)
            return
        if self.fired and self.recovery_step is None:
            counts = ctx.follower_counts()
            top = max(counts, default=0)
            self.min_followers_after = (top if self.min_followers_after < 0
                                        else min(self.min_followers_after, top))
            if top >= ctx.args.recovery_frac * self.followers_pre:
                self.recovery_step = ctx.t - self.fire_at

    def _perturb(self, ctx: RunContext) -> None:
        leader = ctx.opinion_leader()
        self.leader_pre = leader
        if leader < 0:
            self.fired = True     # nothing to perturb; still record the attempt
            return
        self.followers_pre = max(ctx.follower_counts())

        # Push the leader toward a BAD action: the one the reward model does not
        # designate as good in state 0, falling back to a shift on action 0.
        good = ctx.good_actions()
        target = 0 if good is None else int(1 - good[0]) % ctx.args.num_actions
        ctx.push_toward_action(leader, target, self.strength)
        self.fired = True

    def on_run_end(self, ctx: RunContext) -> None:
        self.leader_post = ctx.opinion_leader()

    def measure(self, ctx: RunContext) -> dict[str, Any]:
        # Same keys every run, including when nothing was perturbed — a column
        # that appears only sometimes makes a ragged CSV.
        return {
            "perturb_leader_pre": self.leader_pre,
            "perturb_leader_post": self.leader_post,
            "perturb_followers_pre": self.followers_pre,
            "perturb_min_followers_after": self.min_followers_after,
            "perturb_recovery_step": (-1 if self.recovery_step is None
                                      else self.recovery_step),
            "perturb_recovered": int(self.recovery_step is not None),
            "perturb_leader_changed": int(self.leader_pre != self.leader_post),
        }


class ConvergenceStop(Plugin):
    """
    Stop early once the follower graph has been unchanged for N steps.

    Demonstrates should_stop, and is the piece perturbation_recovery needs for
    its converge-then-shock protocol.
    """

    name = "convergence_stop"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last: Optional[tuple] = None
        self.stable_for = 0
        self.converged_at = -1

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--conv-hold-steps", type=int, default=0,
                            help="Stop after this many steps with an unchanged "
                                 "follower graph. 0 disables early stopping.")

    def on_step(self, ctx: RunContext) -> None:
        current = tuple(ctx.follower_counts())
        if current == self.last:
            self.stable_for += 1
        else:
            self.last, self.stable_for = current, 0
        if self.converged_at < 0 and self._hold(ctx) and self.stable_for >= self._hold(ctx):
            self.converged_at = ctx.t

    def _hold(self, ctx: RunContext) -> int:
        return int(getattr(ctx.args, "conv_hold_steps", 0))

    def should_stop(self, ctx: RunContext) -> bool:
        return self.converged_at > 0

    def measure(self, ctx: RunContext) -> dict[str, Any]:
        return {"converged_at": self.converged_at}
