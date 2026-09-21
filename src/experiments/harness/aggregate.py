"""
Spec-driven aggregation.
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
