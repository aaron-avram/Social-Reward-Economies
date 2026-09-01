"""
Per-step observation plugins.

These are the clearest justification for the `on_step` hook: each needs to see
state as the simulation runs, not just at the end. In the legacy code they were
inline statements inside four separate copies of the stepping loop, which is
why the consensus counters existed only in reputation_status_scaling even
though they would be equally informative in the status sweep.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from src.experiments.harness.plugins import RunContext, RunPlugin


class ConsensusTrackerPlugin(RunPlugin):
    """Counts consensus episodes, where consensus means the top agent holds at
    least `threshold_frac` of the population as followers.

    Three quantities, matching reputation_status_scaling's semantics:
      consensus_step_first -- first step consensus was ever observed (0 if never)
      consensus_step_final -- step at which the LAST consensus episode began
      num_consensus        -- number of episodes after the first
    """

    name = "consensus_tracker"
    columns = ("consensus_step_first", "consensus_step_final", "num_consensus")

    def __init__(self, threshold_frac: float = 0.50) -> None:
        self.threshold_frac = float(threshold_frac)

    def on_start(self, system, ctx: RunContext) -> None:
        ctx.scratch["consensus"] = {
            "first": 0,
            "final": 0,
            "count": 0,
            "current": False,
        }

    def on_step(self, system, ctx: RunContext, step: int, role_updated: bool) -> None:
        state = ctx.scratch["consensus"]
        top = max(len(a.state.followers) for a in system.agents)
        in_consensus = top >= self.threshold_frac * len(system.agents)

        if state["first"] == 0 and in_consensus:
            state["first"] = step
        elif not state["current"] and in_consensus:
            state["final"] = step
            state["count"] += 1
            state["current"] = True
        elif not in_consensus:
            state["current"] = False

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        state = ctx.scratch["consensus"]
        return {
            "consensus_step_first": int(state["first"]),
            "consensus_step_final": int(state["final"]),
            "num_consensus": int(state["count"]),
        }


class ActorRateTrackerPlugin(RunPlugin):
    """Records every agent's actor interaction rate at every step.

    Memory is O(T * N) floats; at 44,000 steps and 50 agents that is ~18 MB per
    run, which is why this is opt-in rather than always on.
    """

    name = "actor_rate_tracker"
    columns = ("leader_actor_rates",)

    def on_start(self, system, ctx: RunContext) -> None:
        ctx.scratch["actor_rates"] = {i: [] for i in range(len(system.agents))}

    def on_step(self, system, ctx: RunContext, step: int, role_updated: bool) -> None:
        rates = ctx.scratch["actor_rates"]
        for i, agent in enumerate(system.agents):
            rates[i].append(agent.state.actor_interaction_rate)

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        rates: Dict[int, List[float]] = ctx.scratch["actor_rates"]
        leader = ctx.summary.leader_id
        if leader > -1:
            series = rates[leader]
        else:
            series = [0] * len(rates[0]) if rates else []
        return {"leader_actor_rates": series}


class FollowerProgressionPlugin(RunPlugin):
    """Emits a downsampled top-follower time series to a side table.

    Contributes no per-run columns -- it writes rows to `ctx.side_tables` -- which
    is the pattern for any diagnostic whose natural shape is a table rather than
    a scalar.
    """

    name = "follower_progression"
    columns = ()

    def __init__(self, table: str = "progression", trace_all: bool = False) -> None:
        self.table = table
        self.trace_all = trace_all

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("progression trace")
        group.add_argument(
            "--trace-seeds",
            type=str,
            default="",
            help='Seeds for which to write sampled top-follower progression, e.g. "0,5".',
        )
        group.add_argument(
            "--trace-every",
            type=int,
            default=100,
            help="Downsample top-follower progression by recording every N timesteps.",
        )

    def _wanted(self, ctx: RunContext) -> bool:
        if self.trace_all:
            return True
        from experiments.harness.cli import parse_csv_ints

        seeds = parse_csv_ints(getattr(ctx.args, "trace_seeds", "") or "")
        return int(ctx.seed) in set(seeds)

    def on_finish(self, system, ctx: RunContext) -> None:
        pass

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        if not self._wanted(ctx):
            return {}

        follower_counts = ctx.summary.follower_counts
        if follower_counts.size == 0:
            return {}

        sample_every = max(1, int(getattr(ctx.args, "trace_every", 100)))
        top_series = np.max(follower_counts, axis=1)
        base = {"mode": ctx.mode, **ctx.cell, "seed": int(ctx.seed)}

        for idx in range(0, len(top_series), sample_every):
            ctx.emit(self.table, {**base, "t": int(idx + 1),
                                  "top_followers": int(top_series[idx])})
        if (len(top_series) - 1) % sample_every != 0:
            ctx.emit(self.table, {**base, "t": int(len(top_series)),
                                  "top_followers": int(top_series[-1])})
        return {}


class AgentTracePlugin(RunPlugin):
    """Emits a terminal per-agent snapshot to a side table."""

    name = "agent_trace"
    columns = ()

    def __init__(self, table: str = "agent_traces") -> None:
        self.table = table

    def _wanted(self, ctx: RunContext) -> bool:
        from experiments.harness.cli import parse_csv_ints

        seeds = parse_csv_ints(getattr(ctx.args, "trace_seeds", "") or "")
        return int(ctx.seed) in set(seeds)

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        if not self._wanted(ctx):
            return {}

        results = ctx.results
        n = int(ctx.args.num_agents)
        nan_row = [float("nan")] * n

        def last(key: str) -> List[float]:
            hist = results.get(key, [])
            return hist[-1] if len(hist) > 0 else nan_row

        est_pu = last("estimated_reward_pu_history")
        est_rep = last("estimated_reward_rep_history")
        est_status = last("estimated_reward_status_history")
        actor_rates = last("actor_interaction_rate_history")

        true_rep = np.asarray(ctx.system._true_reputation().true_reputation, dtype=float)
        base = {"mode": ctx.mode, **ctx.cell, "seed": int(ctx.seed)}

        for agent_id in range(n):
            ctx.emit(self.table, {
                **base,
                "agent_id": int(agent_id),
                "final_role": ctx.summary.final_roles[agent_id],
                "followers": int(ctx.summary.final_followers[agent_id]),
                "estimated_pu": float(est_pu[agent_id]),
                "estimated_rep": float(est_rep[agent_id]),
                "estimated_status": float(est_status[agent_id]),
                "actor_rate": float(actor_rates[agent_id]),
                "true_reputation": float(true_rep[agent_id]),
            })
        return {}
