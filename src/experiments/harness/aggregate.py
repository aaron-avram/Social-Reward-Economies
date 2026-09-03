"""
Spec-driven aggregation.

The legacy `aggregate()` functions were 60-90 lines each of `m1, s1, c1 = ...`
followed by a positional dataclass construction -- a form where inserting a
metric in the middle silently renames every column after it. Here the output
schema is a declared ordered list of columns and the code that fills them is
shared.

Two column kinds:

  Triple(field)  -> mean_<field>, std_<field>, ci95_<field>
  Derived(name)  -> a single column computed from the group

Filtering matters and is explicit. Several legacy metrics dropped sentinel
values before averaging (e.g. time-to-threshold uses -1 for "never reached"),
and one of them -- mean_time_to_90pct_followers -- is therefore CONDITIONAL ON
REACHING. Whether a run reaches is itself a function of gamma and kappa, so
that mean must be read alongside reach_rate, never alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.harness.metrics import mean_std_ci


@dataclass(frozen=True)
class Triple:
    """Emits mean_/std_/ci95_ columns for one per-run field."""

    field: str
    where: Optional[Callable[[Any], bool]] = None
    fallback: Optional[Sequence[float]] = None
    #: Optional override for the column name stem (defaults to `field`).
    stem: Optional[str] = None
    #: Drop non-finite values before averaging. Experiment D's local
    #: `_mean_std_ci` did this unconditionally; A/B/C's did not, and instead
    #: filtered explicitly at each call site with `where=`.
    finite_only: bool = False

    @property
    def columns(self) -> Tuple[str, str, str]:
        s = self.stem or self.field
        return (f"mean_{s}", f"std_{s}", f"ci95_{s}")

    def compute(self, group: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        values = [r[self.field] for r in group]
        if self.where is not None:
            values = [v for v in values if self.where(v)]
        if self.finite_only:
            values = [v for v in values if np.isfinite(float(v))]
        if not values and self.fallback is not None:
            values = list(self.fallback)
        mean, std, ci = mean_std_ci(values)
        c_mean, c_std, c_ci = self.columns
        return {c_mean: mean, c_std: std, c_ci: ci}


@dataclass(frozen=True)
class Derived:
    """Emits a single column computed from the whole group."""

    name: str
    fn: Callable[[Sequence[Dict[str, Any]]], Any]

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.name,)

    def compute(self, group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {self.name: self.fn(group)}


AggregateColumn = Any  # Triple | Derived


def finite(v: Any) -> bool:
    return bool(np.isfinite(v))


def nonneg(v: Any) -> bool:
    return v is not None and v >= 0


def positive(v: Any) -> bool:
    return v is not None and v > 0


def aggregate_columns(spec: Sequence[AggregateColumn]) -> List[str]:
    out: List[str] = []
    for col in spec:
        out.extend(col.columns)
    return out


def aggregate(
    records: Sequence[Dict[str, Any]],
    *,
    group_by: Sequence[str],
    spec: Sequence[AggregateColumn],
) -> List[Dict[str, Any]]:
    """Group `records` by `group_by` and apply `spec` to each group.

    Groups are emitted in sorted key order, matching the legacy behaviour.
    `n_runs` is always emitted immediately after the key columns.
    """
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in records:
        key = tuple(r[k] for k in group_by)
        grouped.setdefault(key, []).append(r)

    rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        group = grouped[key]
        row: Dict[str, Any] = dict(zip(group_by, key))
        row["n_runs"] = len(group)
        for col in spec:
            row.update(col.compute(group))
        rows.append(row)
    return rows
