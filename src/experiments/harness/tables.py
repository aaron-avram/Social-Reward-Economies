"""CSV writing with an explicit, checked column order."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
) -> None:
    """Write `rows` to `path`.

    When `columns` is given it is the authoritative order and every row must
    supply exactly those keys -- a mismatch raises rather than silently dropping
    a metric via `extrasaction="ignore"`. When omitted the first row's key order
    is used, reproducing the legacy behaviour.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(columns) if columns is not None else list(rows[0].keys())

    if columns is not None:
        expected = set(fieldnames)
        for i, row in enumerate(rows):
            got = set(row)
            if got != expected:
                raise ValueError(
                    f"row {i} column mismatch writing {path.name}:\n"
                    f"  missing: {sorted(expected - got)}\n"
                    f"  extra:   {sorted(got - expected)}"
                )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def project(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> list:
    """Select and reorder columns -- the shared shape of every seed-comparison table."""
    return [{c: r[c] for c in columns} for r in rows]
