"""
The single simulation loop.

One `run_single` serves all four experiments. It:

  1. seeds, builds the config, lets each RunPlugin amend it;
  2. constructs the system and fires `on_start`;
  3. steps, firing `before_step` (interventions) and `on_step` (observation)
     around each engine step and driving the async scheduler;
  4. computes the derived `RunSummary` once;
  5. collects each plugin's declared columns into the run record.

The engine still prints during `step()`, which the legacy harnesses worked
around by wrapping the loop in `redirect_stdout`. That workaround lives here,
once, behind `silence_engine`. It is a bug workaround, not a feature: the right
fix is for the engine to route progress through the Recorder / `progress_printer`
rather than stdout, at which point this can go.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout, nullcontext
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from model.system import MultiAgentSystem

from experiments.harness.configspec import make_config
from experiments.harness.metrics import (
    finalize_results,
    leader_series_from_follower_counts,
    role_to_label,
)
from experiments.harness.plugins import RunContext, RunPlugin, RunSummary
from experiments.harness.schedule import RoleUpdateScheduler


def resolve_engine_seed(args, seed: int) -> int:
    """The seed actually handed to the engine's RngBundle.

    Two derivations, because the v2 outputs were produced through the compat
    shim and cannot be reproduced without reproducing its quirk:

      legacy_global -- the v2 path. Harnesses called np.random.seed(s) and never
                       passed a seed to SystemConfig, so compat drew the engine
                       seed from the freshly-seeded GLOBAL stream
                       (np.random.randint). The trajectory is a deterministic
                       function of s, but through one extra layer of laundering.
                       Required for byte-identical parity with committed CSVs.

      direct        -- runtime.seed = s. What the engine's own API intends, what
                       the unit tests assume, and what anyone reading the code
                       would expect. Produces different (equally valid)
                       trajectories.

    The draw must happen after np.random.seed(s) and before the async scheduler
    draws its timers, matching the legacy call order exactly.
    """
    if str(getattr(args, "seed_derivation", "legacy_global")) == "direct":
        return int(seed)
    return int(np.random.randint(0, 2 ** 31 - 1))


def build_summary(ctx: RunContext) -> RunSummary:
    """Derived arrays every experiment shares. Computed once per run."""
    results = ctx.results
    num_agents = int(ctx.args.num_agents)

    follower_counts = np.asarray(results["follower_counts"], dtype=float)
    role_history = np.asarray(results.get("role_label_history", []), dtype=object)
    social_welfare = np.asarray(
        results.get("paper_welfare_followers_only", []), dtype=float
    )

    top_series = (
        follower_counts.max(axis=1) if follower_counts.size else np.array([], dtype=float)
    )
    leader_series = (
        leader_series_from_follower_counts(follower_counts)
        if follower_counts.size
        else np.array([], dtype=int)
    )

    tail_window = (
        min(int(ctx.args.tail_window), len(social_welfare))
        if len(social_welfare) > 0
        else 0
    )

    return RunSummary(
        num_agents=num_agents,
        tail_window=tail_window,
        follower_counts=follower_counts,
        role_history=role_history,
        social_welfare=social_welfare,
        top_follower_series=top_series,
        leader_series=leader_series,
        leader_id=int(results["opinion_leader"]),
        final_roles=[role_to_label(r) for r in results["final_roles"]],
        final_followers=[int(x) for x in results["final_followers"]],
    )


def run_single(
    args,
    *,
    mode: str,
    cell: Dict[str, Any],
    seed: int,
    plugins: Sequence[RunPlugin],
    config_overrides: Optional[Dict[str, Any]] = None,
    num_steps: Optional[int] = None,
    silence_engine: bool = True,
    async_refresh: bool = True,
) -> RunContext:
    """Run one simulation and return its populated context.

    The caller reads `ctx.record` (built by `collect_record`) and
    `ctx.side_tables`. Returning the context rather than a record lets a
    stateful experiment such as D pull out its own diagnostics.
    """
    # Legacy behaviour: the async scheduler's timers come from the global RNG,
    # seeded here. See RoleUpdateScheduler for why and for the opt-out.
    np.random.seed(int(seed))

    ctx = RunContext(args=args, mode=mode, cell=dict(cell), seed=int(seed))

    engine_seed = resolve_engine_seed(args, seed)

    overrides = dict(config_overrides or {})
    overrides.update(cell)
    config = make_config(args, mode=mode, seed=engine_seed, overrides=overrides)
    for plugin in plugins:
        config = plugin.configure(config, ctx)

    system = MultiAgentSystem(config)
    ctx.system = system

    scheduler = RoleUpdateScheduler.build(args, seed=seed, refresh=async_refresh)
    ctx.scratch["scheduler"] = scheduler

    for plugin in plugins:
        plugin.on_start(system, ctx)

    total_steps = int(num_steps if num_steps is not None else args.num_steps)
    stop = {"now": False}
    ctx.scratch["request_stop"] = lambda: stop.__setitem__("now", True)
    guard = redirect_stdout(io.StringIO()) if silence_engine else nullcontext()

    with guard:
        for i in range(total_steps):
            next_step = int(system.time_step) + 1
            for plugin in plugins:
                plugin.before_step(system, ctx, next_step)

            system.step()

            role_updated = scheduler.after_step(system)

            step = i + 1
            for plugin in plugins:
                plugin.on_step(system, ctx, step, role_updated)

            if stop["now"]:
                break

        for plugin in plugins:
            plugin.on_finish(system, ctx)

        ctx.results = finalize_results(system)
        ctx.results.update(ctx.scratch.get("result_extras", {}))

    ctx.summary = build_summary(ctx)
    return ctx


def collect_record(ctx: RunContext, plugins: Sequence[RunPlugin]) -> Dict[str, Any]:
    """Merge key columns with every plugin's declared measurement columns."""
    record: Dict[str, Any] = ctx.key_columns()
    for plugin in plugins:
        for key, value in plugin.measure(ctx).items():
            if key in record:
                raise RuntimeError(
                    f"plugin {plugin.name!r} overwrote column {key!r}; "
                    "column ownership should have caught this at startup"
                )
            record[key] = value
    return record
