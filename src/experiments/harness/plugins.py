"""
Plugin protocols.

There are two extension lifetimes and they are deliberately separate types. A
single flat `Plugin` class with a dozen optional methods would hide which hooks
fire when, and would let a sweep-level concern accidentally reach into a run.

RunPlugin  -- lives for one simulation. Adds CLI flags, mutates the config,
              observes steps, and contributes columns to that run's record.
SweepPlugin -- lives for the whole grid. Adds CLI flags, contributes aggregate
              columns, and emits figures and side tables.

Column ownership is checked at Experiment construction: two plugins may not
declare the same column, and the union of declared columns must match the
experiment's record schema exactly. That turns a whole class of silent
wiring bug -- a flag declared but never threaded through, a metric computed but
never written -- into an ImportError at startup.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


# ------------------------------------------------------------- contexts ---

@dataclass
class RunSummary:
    """Derived arrays every experiment needs, computed once by the runner."""

    num_agents: int
    tail_window: int
    follower_counts: np.ndarray          # (T, N)
    role_history: np.ndarray             # (T, N) of role enums, may be empty
    social_welfare: np.ndarray           # (T,) paper welfare, followers only
    top_follower_series: np.ndarray      # (T,)
    leader_series: np.ndarray            # (T,) lowest-id tie-break
    leader_id: int                       # terminal opinion leader, -1 if none
    final_roles: List[str]
    final_followers: List[int]


@dataclass
class RunContext:
    """Everything a RunPlugin can see about one simulation."""

    args: argparse.Namespace
    mode: str
    cell: Dict[str, Any]                 # axis name -> value for this grid cell
    seed: int
    system: Any = None
    results: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[RunSummary] = None

    # Free-form space for a plugin to carry state from on_step to measure.
    scratch: Dict[str, Any] = field(default_factory=dict)
    # Auxiliary row tables keyed by logical name; the runner concatenates these
    # across runs and the experiment decides which get written to disk.
    side_tables: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def emit(self, table: str, row: Dict[str, Any]) -> None:
        self.side_tables.setdefault(table, []).append(row)

    def key_columns(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"mode": self.mode}
        out.update(self.cell)
        out["seed"] = int(self.seed)
        return out


@dataclass
class SweepContext:
    """Everything a SweepPlugin can see after the whole grid has run."""

    args: argparse.Namespace
    mode: str
    axis_names: Tuple[str, ...]
    seeds: Tuple[int, ...]
    records: List[Dict[str, Any]]
    aggregates: List[Dict[str, Any]]
    side_tables: Dict[str, List[Dict[str, Any]]]
    output_dir: Path
    #: Escape hatch: the finished RunContexts, in execution order. Needed when a
    #: figure requires per-run time series rather than tabular rows -- Experiment
    #: D's per-seed trajectory plots are the motivating case. Prefer
    #: `side_tables` where the data is genuinely rows; reaching in here couples a
    #: sweep plugin to run internals.
    run_contexts: List[Any] = field(default_factory=list)


# -------------------------------------------------------------- plugins ---

class RunPlugin:
    """Base class for per-run extensions. Every hook is optional."""

    #: Stable identifier, used in error messages and ordering.
    name: str = "run_plugin"

    #: Columns this plugin contributes to the per-run record, in output order.
    columns: Tuple[str, ...] = ()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Declare CLI flags this plugin needs."""

    def configure(self, config, ctx: RunContext):
        """Return a (possibly modified) SystemConfig before the system is built."""
        return config

    def on_start(self, system, ctx: RunContext) -> None:
        """Called once after the system is constructed, before the first step."""

    def before_step(self, system, ctx: RunContext, next_step: int) -> None:
        """Called before each `system.step()`. This is where interventions land."""

    def on_step(self, system, ctx: RunContext, step: int, role_updated: bool) -> None:
        """Called after each `system.step()` and after any role update."""

    def on_finish(self, system, ctx: RunContext) -> None:
        """Called once after the loop, before `summary` is computed."""

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        """Return this plugin's columns for the finished run."""
        return {}


class SweepPlugin:
    """Base class for whole-grid extensions. Every hook is optional."""

    name: str = "sweep_plugin"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Declare CLI flags this plugin needs."""

    def figures(self, ctx: SweepContext) -> None:
        """Emit plots and any extra CSVs."""


class MetricsPlugin(RunPlugin):
    """Adapter turning a plain function into a RunPlugin.

    Most experiments only need to declare metrics. Rather than making each one
    subclass, they pass an ordered column tuple and a function over RunContext.
    The column tuple is the contract; the function must return exactly those
    keys, and the runner checks that it does.
    """

    def __init__(self, name: str, columns: Sequence[str], fn) -> None:
        self.name = name
        self.columns = tuple(columns)
        self._fn = fn

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        out = self._fn(ctx)
        missing = set(self.columns) - set(out)
        extra = set(out) - set(self.columns)
        if missing or extra:
            raise RuntimeError(
                f"MetricsPlugin {self.name!r} contract violation: "
                f"missing={sorted(missing)} unexpected={sorted(extra)}"
            )
        return out


def check_column_ownership(
    run_plugins: Sequence[RunPlugin],
    key_columns: Sequence[str],
    record_columns: Sequence[str],
) -> None:
    """Fail loudly at construction if the record schema and plugins disagree."""
    owner: Dict[str, str] = {c: "<key>" for c in key_columns}
    for plugin in run_plugins:
        for col in plugin.columns:
            if col in owner:
                raise ValueError(
                    f"column {col!r} declared by both {owner[col]!r} and {plugin.name!r}"
                )
            owner[col] = plugin.name

    declared = set(owner)
    expected = set(record_columns)
    if declared != expected:
        raise ValueError(
            "record schema mismatch:\n"
            f"  declared but not in schema: {sorted(declared - expected)}\n"
            f"  in schema but unclaimed:    {sorted(expected - declared)}"
        )
