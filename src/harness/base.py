"""
Unified experiment harness.

An experiment declares WHAT to sweep and WHAT to measure. This module owns
everything else: argument parsing, the grid, seeding, the run loop, aggregation,
and CSV output. That division is what the four original harnesses got wrong —
19 helper functions were duplicated across three or more of them.

    class MyExperiment(Experiment):
        name = "my_experiment"

        def axes(self, args):
            return [Axis("gamma", parse_floats(args.gammas),
                         apply=lambda cfg, v: with_algorithm(cfg, gamma=v))]

        def measure(self, ctx):
            return {"max_followers": max(ctx.follower_counts())}

    if __name__ == "__main__":
        run_cli(MyExperiment())

SEEDING
-------
Seeds come from SeedSequence(base).spawn(n), and the SAME seed set is used at
every grid point. That is deliberate: it makes comparisons across grid points
PAIRED, so a difference between two gamma values reflects gamma rather than
different activation draws. The old harnesses used np.random.seed(i) with
sequential ints, which does not give this.
"""

from __future__ import annotations

import csv
import itertools
import math
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from model.config import (
    ActorRateDriverMode, AlgorithmParams, Dimensions, Eq9Mode, LeaderUpdateMode,
    RewardModelKind, RewardParams, RuntimeParams, ScheduleParams, SystemConfig,
    TrackingMode,
)
from model.instrumentation import FullRecorder, NullRecorder
from model.system import MultiAgentSystem

from harness.plugin import Axis, Plugin, RunContext


# ============================================================================
# Config helpers — the frozen param groups need nested replace()
# ============================================================================

def with_algorithm(cfg: SystemConfig, **kw) -> SystemConfig:
    return replace(cfg, algorithm=replace(cfg.algorithm, **kw))


def with_reward(cfg: SystemConfig, **kw) -> SystemConfig:
    return replace(cfg, reward=replace(cfg.reward, **kw))


def with_dims(cfg: SystemConfig, **kw) -> SystemConfig:
    return replace(cfg, dims=replace(cfg.dims, **kw))


def with_runtime(cfg: SystemConfig, **kw) -> SystemConfig:
    return replace(cfg, runtime=replace(cfg.runtime, **kw))


def with_schedule(cfg: SystemConfig, **kw) -> SystemConfig:
    return replace(cfg, schedule=replace(cfg.schedule, **kw))


# ============================================================================
# Argument parsing helpers, shared by every experiment
# ============================================================================

def parse_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def parse_strings(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


# ============================================================================
# Experiment
# ============================================================================

class Experiment:
    """
    Subclass and override. Only `name` and `measure` are required; an
    experiment with no axes runs a single grid point.
    """

    name: str = "experiment"
    plugins: Sequence[Plugin] = ()

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Arguments beyond the common set. The base adds the common ones."""

    def axes(self, args: Namespace) -> list[Axis]:
        """Sweep dimensions. Empty means one grid point."""
        return []

    def base_config(self, args: Namespace) -> SystemConfig:
        """
        The config before any axis is applied. The default reads the common
        arguments; override to change defaults, and call super() first.
        """
        return default_config(args)

    def measure(self, ctx: RunContext) -> dict[str, Any]:
        """Per-run outcome columns. Same keys on every run."""
        raise NotImplementedError


# ============================================================================
# Common configuration
# ============================================================================

def add_common_arguments(parser: ArgumentParser) -> None:
    """The 24 arguments every one of the four original harnesses accepted."""
    g = parser.add_argument_group("simulation")
    g.add_argument("--num-agents", type=int, default=6)
    g.add_argument("--num-states", type=int, default=3)
    g.add_argument("--num-actions", type=int, default=2)
    g.add_argument("--num-steps", type=int, default=2000)
    g.add_argument("--mode", choices=["static", "async"], default="static")

    g = parser.add_argument_group("algorithm")
    g.add_argument("--gamma", type=float, default=2.0)
    g.add_argument("--kappa", type=float, default=2.0)
    g.add_argument("--c-threshold", type=float, default=0.1)
    g.add_argument("--B-R", type=float, default=0.8)
    g.add_argument("--B-F", type=float, default=0.6)
    g.add_argument("--delta", type=float, default=0.1)
    g.add_argument("--initial-actor-rate", type=float, default=0.7)
    g.add_argument("--initial-participant-rate", type=float, default=0.7)

    g = parser.add_argument_group("reward")
    g.add_argument("--reward-model", default="simple_preferred_action",
                   choices=[k.value for k in RewardModelKind])
    g.add_argument("--reward-base-mu", type=float, default=0.5)
    g.add_argument("--reward-base-sigma", type=float, default=0.08)
    g.add_argument("--reward-agent-sigma", type=float, default=0.03)
    g.add_argument("--reward-clip-min", type=float, default=0.01)
    g.add_argument("--reward-clip-max", type=float, default=2.5)

    g = parser.add_argument_group("role updates")
    g.add_argument("--role-update-base-interval", type=int, default=50)
    g.add_argument("--fixed-role-update-interval", action="store_true")
    g.add_argument("--role-update-s0", type=int, default=0)
    g.add_argument("--role-update-epochs", type=str, default="")
    g.add_argument("--async-role-update-prob", type=float, default=None,
                   help="Per-agent per-step probability of reevaluating roles "
                        "in async mode. Default: 1/role_update_base_interval.")

    g = parser.add_argument_group("modes")
    g.add_argument("--eq9-averaging-mode", default="participants_only",
                   choices=[m.value for m in Eq9Mode])
    g.add_argument("--leader-update-mode", default="participants_only_post_eq9",
                   choices=[m.value for m in LeaderUpdateMode])
    g.add_argument("--actor-rate-driver-mode", default="standard",
                   choices=[m.value for m in ActorRateDriverMode])
    g.add_argument("--actor-rate-status-override-min-followers", type=int,
                   default=10)
    g.add_argument("--force-all-active-debug", action="store_true")
    g.add_argument("--numpy-fast-path", action="store_true",
                   help="Accepted and IGNORED. The two phase-4 implementations "
                        "merged into one, so there is no longer a fast path to "
                        "select. Kept so existing invocations keep working.")

    g = parser.add_argument_group("run")
    g.add_argument("--seeds", type=int, default=10)
    g.add_argument("--seed-base", type=int, default=20260101,
                   help="Root of the SeedSequence. The same seed set is used at "
                        "every grid point, making cross-point comparisons paired.")
    g.add_argument("--selected-seeds", type=str, default="",
                   help="Comma-separated replicate INDICES (0-based) to run, "
                        "e.g. '3,7'. Not seed values: seeds come from a "
                        "SeedSequence and are not consecutive, unlike v1's "
                        "--seed-start.")
    g.add_argument("--tail-window", type=int, default=200)
    g.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    g.add_argument("--output-dir", type=str,
                   default=str(Path.cwd() / "outputs"))


def default_config(args: Namespace) -> SystemConfig:
    """Translate the common arguments into a SystemConfig."""
    # In async mode agents reevaluate on independent clocks, so the scheduled
    # role update must never fire; push it past the horizon.
    interval = (args.num_steps + 1_000_000 if args.mode == "async"
                else args.role_update_base_interval)
    epochs = parse_ints(args.role_update_epochs) if args.role_update_epochs else []

    return SystemConfig(
        dims=Dimensions(num_agents=args.num_agents, num_states=args.num_states,
                        num_actions=args.num_actions),
        algorithm=AlgorithmParams(
            gamma=args.gamma, kappa=args.kappa, c_threshold=args.c_threshold,
            B_R=args.B_R, B_F=args.B_F, delta=args.delta,
            initial_actor_interaction_rate=args.initial_actor_rate,
            initial_participant_interaction_rate=args.initial_participant_rate,
            actor_rate_status_override_min_followers=(
                args.actor_rate_status_override_min_followers),
            actor_rate_driver_mode=ActorRateDriverMode(args.actor_rate_driver_mode),
            eq9_averaging_mode=Eq9Mode(args.eq9_averaging_mode),
            leader_update_mode=LeaderUpdateMode(args.leader_update_mode),
        ),
        reward=RewardParams(
            kind=RewardModelKind(args.reward_model),
            base_mu=args.reward_base_mu, base_sigma=args.reward_base_sigma,
            agent_sigma=args.reward_agent_sigma,
            clip_min=args.reward_clip_min, clip_max=args.reward_clip_max,
        ),
        runtime=RuntimeParams(
            seed=0,                       # set per run
            num_time_steps=args.num_steps,
            tracking_mode=TrackingMode(args.tracking_mode),
            force_all_active_debug=args.force_all_active_debug,
        ),
        schedule=ScheduleParams(
            role_update_s0=args.role_update_s0,
            role_update_base_interval=interval,
            fixed_role_update_interval=args.fixed_role_update_interval,
            role_update_epochs=[] if args.mode == "async" else epochs,
        ),
    )


# ============================================================================
# The grid
# ============================================================================

@dataclass(frozen=True)
class GridPoint:
    """One combination of axis values, plus the config it produces."""
    values: dict[str, Any]
    config: SystemConfig
    plugin_settings: dict[int, dict[str, Any]]   # plugin index -> {attr: value}

    def label(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.values.items()) or "(single)"


def build_grid(experiment: Experiment, args: Namespace) -> list[GridPoint]:
    """
    Cartesian product of the experiment's axes AND every plugin's axes.

    Including plugin axes is the point of the design: it makes
    "gamma x perturbation_strength" one grid rather than two experiments.
    """
    exp_axes = list(experiment.axes(args))
    plugin_axes: list[tuple[int, Axis]] = [
        (i, axis)
        for i, plugin in enumerate(experiment.plugins)
        for axis in plugin.axes(args)
    ]

    names = [a.name for a in exp_axes] + [a.name for _, a in plugin_axes]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate axis names: {sorted(duplicates)}")

    all_axes = [(None, a) for a in exp_axes] + plugin_axes
    if not all_axes:
        return [GridPoint({}, experiment.base_config(args), {})]

    points = []
    for combo in itertools.product(*[a.values for _, a in all_axes]):
        cfg = experiment.base_config(args)
        values: dict[str, Any] = {}
        settings: dict[int, dict[str, Any]] = {}

        for (plugin_idx, axis), value in zip(all_axes, combo):
            values[axis.name] = value
            if axis.apply is not None:
                cfg = axis.apply(cfg, value)
            else:
                settings.setdefault(plugin_idx, {})[axis.store] = value

        for plugin in experiment.plugins:
            cfg = plugin.configure(args, cfg)

        points.append(GridPoint(values, cfg, settings))
    return points


def resolve_seeds(args: Namespace) -> list[int]:
    """
    One independent seed per replicate, shared across grid points.

    SeedSequence.spawn gives streams that are independent by construction,
    unlike sequential integer seeds.
    """
    root = np.random.SeedSequence(int(args.seed_base))
    seeds = [int(child.generate_state(1)[0] % (2**31 - 1))
             for child in root.spawn(int(args.seeds))]

    chosen = getattr(args, "selected_seeds", "")
    if chosen:
        # Indices, not values: the seeds are not consecutive, so selecting by
        # value would be unusable. Out-of-range indices are an error rather than
        # a silent no-op, since a typo would otherwise run the whole sweep.
        idx = parse_ints(chosen)
        bad = [i for i in idx if not 0 <= i < len(seeds)]
        if bad:
            raise ValueError(f"--selected-seeds index out of range: {bad} "
                             f"(have {len(seeds)} replicates)")
        seeds = [seeds[i] for i in idx]
    return seeds


# ============================================================================
# Running
# ============================================================================

def run_one(experiment: Experiment, args: Namespace, point: GridPoint,
            seed: int) -> dict[str, Any]:
    """Execute a single (grid point, seed) run and return its CSV row."""
    cfg = with_runtime(point.config, seed=seed)

    recorder = NullRecorder()
    system = MultiAgentSystem(cfg, rec=recorder)
    ctx = RunContext(system=system, args=args, point=dict(point.values), seed=seed)

    plugins = list(experiment.plugins)
    for i, plugin in enumerate(plugins):
        plugin.reset()
        for attr, value in point.plugin_settings.get(i, {}).items():
            setattr(plugin, attr, value)
        plugin.on_run_start(ctx)

    rng = np.random.default_rng(seed)      # for async role-update scheduling
    prob = (args.async_role_update_prob
            if args.async_role_update_prob is not None
            else 1.0 / max(1, args.role_update_base_interval))

    for t in range(1, args.num_steps + 1):
        system.step()
        if args.mode == "async":
            candidates = [i for i in range(cfg.dims.num_agents)
                          if rng.random() < prob]
            if candidates:
                system.update_roles(candidates)
        ctx.t = t
        for plugin in plugins:
            plugin.on_step(ctx)
        if any(plugin.should_stop(ctx) for plugin in plugins):
            break

    for plugin in plugins:
        plugin.on_run_end(ctx)

    row: dict[str, Any] = {**point.values, "seed": seed, "steps_run": ctx.t}
    row.update(experiment.measure(ctx))

    for plugin in plugins:
        extra = plugin.measure(ctx)
        clash = set(extra) & set(row)
        if clash:
            raise ValueError(
                f"plugin {plugin.name!r} measure() collides on {sorted(clash)}")
        row.update(extra)
    return row


def run_sweep(experiment: Experiment, args: Namespace) -> list[dict[str, Any]]:
    grid = build_grid(experiment, args)
    seeds = resolve_seeds(args)
    rows = []
    for point in grid:
        for seed in seeds:
            rows.append(run_one(experiment, args, point, seed))
        print(f"[done] {point.label()} ({len(seeds)} seeds)")
    return rows


# ============================================================================
# Aggregation and output
# ============================================================================

def mean_std_ci(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean, sample sd, and half-width of the 95% CI."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, sd, ci


def aggregate(rows: Sequence[dict], axis_names: Sequence[str]) -> list[dict]:
    """Collapse seeds within each grid point into mean/sd/ci per numeric column."""
    if not rows:
        return []
    metric_cols = [
        c for c in rows[0]
        if c not in set(axis_names) | {"seed"}
        and all(isinstance(r.get(c), (int, float, bool)) for r in rows)
    ]
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[a] for a in axis_names), []).append(row)

    out = []
    for key, group in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        agg = dict(zip(axis_names, key))
        agg["n_seeds"] = len(group)
        for col in metric_cols:
            mean, sd, ci = mean_std_ci([r[col] for r in group])
            agg[f"mean_{col}"] = mean
            agg[f"std_{col}"] = sd
            agg[f"ci95_{col}"] = ci
        out.append(agg)
    return out


def write_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


# ============================================================================
# Entry point
# ============================================================================

def build_parser(experiment: Experiment) -> ArgumentParser:
    parser = ArgumentParser(description=experiment.name)
    add_common_arguments(parser)
    experiment.add_arguments(parser)
    for plugin in experiment.plugins:
        plugin.add_arguments(parser)
    return parser


def run_cli(experiment: Experiment, argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(experiment).parse_args(argv)
    rows = run_sweep(experiment, args)

    axis_names = [a.name for a in experiment.axes(args)]
    axis_names += [a.name for p in experiment.plugins for a in p.axes(args)]

    outdir = Path(args.output_dir)
    write_csv(rows, outdir / f"{experiment.name}_runs_{args.mode}.csv")
    write_csv(aggregate(rows, axis_names),
              outdir / f"{experiment.name}_aggregate_{args.mode}.csv")
