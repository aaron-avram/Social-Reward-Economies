"""
Sweep axes and grid enumeration.

The four legacy harnesses each hard-coded their own nested loop:

    for reward_model: for num_states: for seed:      # Experiment A
    for gamma:        for kappa:      for seed:      # Experiments B / C
    for seed:                                        # Experiment D

Those are the same loop with different axes. Making the axes *data* rather than
control flow is what lets one runner serve all four, and it is what makes the
per-run CSV key columns, the aggregation grouping, and the figure faceting fall
out of a single declaration instead of being restated three times each.

Iteration order is the Cartesian product in axis order, seeds innermost. That
reproduces the legacy row order exactly, which is what makes byte-identical
parity against the committed baselines a meaningful test.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple


@dataclass(frozen=True)
class Axis:
    """One swept dimension.

    `name` is the CSV column name and the `make_config` keyword. `values` is the
    ordered list of settings; the outermost axis is the first in `Grid.axes`.
    """

    name: str
    values: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Axis.name must be non-empty")
        if len(self.values) == 0:
            raise ValueError(f"Axis {self.name!r} has no values")

    @classmethod
    def of(cls, name: str, values: Sequence[Any]) -> "Axis":
        return cls(name=name, values=tuple(values))


@dataclass(frozen=True)
class Grid:
    """A Cartesian product of axes, replicated over seeds.

    A zero-axis grid is legal and is exactly what Experiment D needs: one cell,
    swept over seeds only.
    """

    axes: Tuple[Axis, ...]
    seeds: Tuple[int, ...]

    def __post_init__(self) -> None:
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate axis names: {names}")
        if len(self.seeds) == 0:
            raise ValueError("Grid needs at least one seed")

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(a.name for a in self.axes)

    def cells(self) -> Iterator[Dict[str, Any]]:
        if not self.axes:
            yield {}
            return
        for combo in itertools.product(*(a.values for a in self.axes)):
            yield dict(zip(self.axis_names, combo))

    def runs(self) -> Iterator[Tuple[Dict[str, Any], int]]:
        """(cell, seed) pairs in execution order: cells outer, seeds inner."""
        for cell in self.cells():
            for seed in self.seeds:
                yield cell, seed

    def __len__(self) -> int:
        n_cells = 1
        for a in self.axes:
            n_cells *= len(a.values)
        return n_cells * len(self.seeds)


def format_num(value: float) -> str:
    """Filename-safe number formatting (verbatim from reputation_status_scaling)."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p").replace("-", "m")


def cell_tag(cell: Dict[str, Any]) -> str:
    """Filename-safe tag for a grid cell, e.g. 'g5_k2' or 'shared_base_gaussian_S10'."""
    parts: List[str] = []
    for name, value in cell.items():
        head = name[0] if name else "x"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{head}{format_num(float(value))}")
        else:
            parts.append(str(value))
    return "_".join(parts)


def cell_label(cell: Dict[str, Any]) -> str:
    """Human-readable label for legends and titles."""
    parts = []
    for name, value in cell.items():
        if isinstance(value, float):
            parts.append(f"{name}={value:g}")
        else:
            parts.append(f"{name}={value}")
    return ", ".join(parts)
