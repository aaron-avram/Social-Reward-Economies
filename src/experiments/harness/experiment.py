"""
The Experiment declaration and its `main()`.

An experiment is now a value, not a script: axes, plugins, a record schema, an
aggregation spec, and output naming. `Experiment.main()` is the same driver for
all of them.

Declaring `record_columns` explicitly may look redundant next to the plugins
that fill them, but it is the point: it is the CSV contract, checked against
plugin ownership at construction. A metric that no plugin claims, or a plugin
column absent from the schema, is an error at startup rather than a column that
silently vanishes from a sweep that took six hours.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from experiments.harness.aggregate import AggregateColumn, aggregate, aggregate_columns
from experiments.harness.axes import Axis, Grid
from experiments.harness.cli import resolve_seeds
from experiments.harness.plugins import (
    RunContext,
    RunPlugin,
    SweepContext,
    SweepPlugin,
    check_column_ownership,
)
from experiments.harness.runner import collect_record, run_single
from experiments.harness.tables import write_csv


@dataclass
class Experiment:
    """A complete experiment definition."""

    #: Short slug used as the CSV/PNG filename prefix, e.g. "pu_scaling".
    name: str
    description: str

    #: Builds the argument parser. Typically calls cli.add_core_arguments then
    #: adds experiment-specific flags.
    build_parser: Callable[[], argparse.ArgumentParser]

    #: Maps parsed args to the sweep axes (seeds are handled separately).
    build_axes: Callable[[argparse.Namespace], Sequence[Axis]]

    #: Per-run plugins, in the order their columns appear in the CSV.
    run_plugins: Sequence[RunPlugin] = field(default_factory=tuple)
    sweep_plugins: Sequence[SweepPlugin] = field(default_factory=tuple)

    #: Authoritative per-run CSV column order, including keys.
    record_columns: Sequence[str] = field(default_factory=tuple)

    #: Aggregation output, in column order (after the keys and n_runs).
    aggregate_spec: Sequence[AggregateColumn] = field(default_factory=tuple)

    #: Extra SystemConfig fields fixed by this experiment (e.g. gamma=0 for A).
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    #: Columns of the seed-comparison table; omitted if empty.
    seed_comparison_columns: Sequence[str] = field(default_factory=tuple)
    #: Sort key for the seed-comparison table.
    seed_comparison_sort: Optional[Sequence[str]] = None

    #: Side tables to write, mapping table name -> filename stem.
    side_table_files: Dict[str, str] = field(default_factory=dict)

    #: One-line progress string per finished run.
    progress_line: Optional[Callable[[Dict[str, Any]], str]] = None

    #: Attribute on `args` holding the step horizon. Experiment D uses
    #: `num_steps_max` because it stops on a criterion, not at the horizon.
    steps_attr: str = "num_steps"

    #: Optional override for the output directory, e.g. a per-run subdirectory
    #: stamped with the parameters. Receives parsed args.
    output_dir_fn: Optional[Callable[[argparse.Namespace], Path]] = None

    #: Optional override for the filename stem (defaults to `name`), for
    #: experiments that expose the prefix as a CLI flag.
    file_stem_fn: Optional[Callable[[argparse.Namespace], str]] = None

    #: Optional per-run scheduler flag; see RoleUpdateScheduler.refresh.
    async_refresh: bool = True

    # -- wiring ------------------------------------------------------------

    def parser(self) -> argparse.ArgumentParser:
        p = self.build_parser()
        for plugin in list(self.run_plugins) + list(self.sweep_plugins):
            plugin.add_arguments(p)
        return p

    def validate(self, axis_names: Sequence[str]) -> None:
        key_columns = ["mode", *axis_names, "seed"]
        check_column_ownership(self.run_plugins, key_columns, self.record_columns)

    @property
    def aggregate_columns(self) -> List[str]:
        return aggregate_columns(self.aggregate_spec)

    # -- driver ------------------------------------------------------------

    def main(self, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = self.parser().parse_args(argv)
        output_dir = (
            self.output_dir_fn(args) if self.output_dir_fn is not None
            else Path(args.output_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.file_stem_fn(args) if self.file_stem_fn is not None else self.name

        axes = tuple(self.build_axes(args))
        seeds = tuple(resolve_seeds(args))
        grid = Grid(axes=axes, seeds=seeds)
        self.validate(grid.axis_names)

        agg_keys = ["mode", *grid.axis_names]
        agg_columns = [*agg_keys, "n_runs", *self.aggregate_columns]

        print("#" * 72)
        print(f"{self.description}")
        print(f"mode={args.mode}")
        for axis in axes:
            print(f"{axis.name}={list(axis.values)}")
        print(f"grid cells={len(list(grid.cells()))}, seeds={len(seeds)}, runs={len(grid)}")
        print("#" * 72)

        records: List[Dict[str, Any]] = []
        side_tables: Dict[str, List[Dict[str, Any]]] = {}
        contexts: List[RunContext] = []

        total = len(grid)
        started = time.time()
        for job, (cell, seed) in enumerate(grid.runs(), start=1):
            ctx = run_single(
                args,
                mode=args.mode,
                cell=cell,
                seed=seed,
                plugins=self.run_plugins,
                config_overrides=self.config_overrides,
                num_steps=int(getattr(args, self.steps_attr)),
                async_refresh=self.async_refresh,
            )
            record = collect_record(ctx, self.run_plugins)
            records.append(record)
            contexts.append(ctx)
            for table, rows in ctx.side_tables.items():
                side_tables.setdefault(table, []).extend(rows)

            if self.progress_line is not None:
                print(f"[{job:03d}/{total:03d}] {self.progress_line(record)}")
            else:
                print(f"[{job:03d}/{total:03d}] {cell} seed={seed}")

        aggregates = aggregate(
            records, group_by=agg_keys, spec=self.aggregate_spec
        )

        run_csv = output_dir / f"{stem}_runs_{args.mode}.csv"
        agg_csv = output_dir / f"{stem}_aggregate_{args.mode}.csv"
        write_csv(run_csv, records, columns=self.record_columns)
        write_csv(agg_csv, aggregates, columns=agg_columns)
        written = [run_csv, agg_csv]

        if self.seed_comparison_columns:
            rows = records
            if self.seed_comparison_sort:
                keys = list(self.seed_comparison_sort)
                rows = sorted(rows, key=lambda r: tuple(r[k] for k in keys))
            seed_rows = [{c: r[c] for c in self.seed_comparison_columns} for r in rows]
            seed_csv = output_dir / f"{stem}_seed_comparison_{args.mode}.csv"
            write_csv(seed_csv, seed_rows, columns=self.seed_comparison_columns)
            written.append(seed_csv)

        for table, file_stem in self.side_table_files.items():
            rows = side_tables.get(table, [])
            if not rows:
                continue
            path = output_dir / f"{file_stem}_{args.mode}.csv"
            write_csv(path, rows)
            written.append(path)

        sweep_ctx = SweepContext(
            args=args,
            mode=args.mode,
            axis_names=grid.axis_names,
            seeds=seeds,
            records=records,
            aggregates=aggregates,
            side_tables=side_tables,
            output_dir=output_dir,
            run_contexts=contexts,
        )
        for plugin in self.sweep_plugins:
            plugin.figures(sweep_ctx)

        elapsed = time.time() - started
        print()
        for path in written:
            print(f"Wrote {path}")
        print(f"Wrote plots to: {output_dir}")
        print(f"Completed {total} runs in {elapsed:.1f}s")

        return {
            "args": args,
            "records": records,
            "aggregates": aggregates,
            "side_tables": side_tables,
            "contexts": contexts,
            "output_dir": output_dir,
        }
