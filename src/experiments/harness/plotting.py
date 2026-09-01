"""
Shared plotting.

`plot_metric`, the errorbar helper, and the gamma-kappa heatmap were each
reimplemented per harness with slightly different faceting. Here the faceting is
a parameter: `line_by` names the record fields that separate one line from
another, `x_field` names the abscissa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig_path: Path, *, dpi: int = 180, tight_bbox: bool = False) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    if tight_bbox:
        plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_metric(
    aggregate_rows: Sequence[Dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    ylabel: str,
    output_file: Path,
    line_by: Sequence[str] = ("mode",),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    label_fmt: Optional[str] = None,
    figsize: Tuple[float, float] = (6.8, 4.5),
    dpi: int = 180,
) -> None:
    """One line per distinct combination of `line_by`, x = `x_field`."""
    if not aggregate_rows:
        return

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in aggregate_rows:
        key = tuple(row[f] for f in line_by)
        groups.setdefault(key, []).append(row)

    plt.figure(figsize=figsize)
    for key, rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rows = sorted(rows, key=lambda r: r[x_field])
        xs = np.array([r[x_field] for r in rows], dtype=float)
        ys = np.array([r[y_field] for r in rows], dtype=float)
        if label_fmt is not None:
            label = label_fmt.format(**dict(zip(line_by, key)))
        else:
            label = " | ".join(str(k) for k in key)
        plt.plot(xs, ys, "-o", linewidth=1.8, label=label)

    plt.xlabel(xlabel if xlabel is not None else x_field.replace("_", " "))
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.25)
    if len(groups) > 1:
        plt.legend()
    _save(output_file, dpi=dpi)


def plot_errorbar(
    aggregate_rows: Sequence[Dict[str, Any]],
    *,
    x_field: str,
    mean_field: str,
    err_field: Optional[str],
    ylabel: str,
    output_file: Path,
    line_by: Sequence[str] = (),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    cmap_name: str = "viridis",
    figsize: Tuple[float, float] = (7.0, 4.6),
    dpi: int = 220,
) -> None:
    """Errorbar plot, one series per `line_by` combination, viridis-coloured.

    With `line_by=("gamma",)` this is the multi-gamma figure: each gamma gets its
    own colour from the viridis ramp so the ordering of gamma is readable from
    the colour alone.
    """
    if not aggregate_rows:
        return

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in aggregate_rows:
        key = tuple(row[f] for f in line_by)
        groups.setdefault(key, []).append(row)

    keys = sorted(groups.keys())
    cmap = plt.get_cmap(cmap_name)
    colors = (
        [cmap(i / max(1, len(keys) - 1)) for i in range(len(keys))]
        if len(keys) > 1
        else [cmap(0.5)]
    )

    plt.figure(figsize=figsize)
    for color, key in zip(colors, keys):
        rows = sorted(groups[key], key=lambda r: r[x_field])
        xs = np.array([r[x_field] for r in rows], dtype=float)
        ys = np.array([r[mean_field] for r in rows], dtype=float)
        errs = (
            np.array([r[err_field] for r in rows], dtype=float)
            if err_field is not None
            else None
        )
        label = ", ".join(f"{n}={v:g}" if isinstance(v, float) else f"{n}={v}"
                          for n, v in zip(line_by, key)) if line_by else None
        plt.errorbar(
            xs, ys, yerr=errs, fmt="-o", linewidth=1.8,
            capsize=3, color=color, label=label,
        )

    plt.xlabel(xlabel if xlabel is not None else x_field)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.25)
    if line_by and len(keys) > 1:
        plt.legend(fontsize=8)
    _save(output_file, dpi=dpi, tight_bbox=True)


def plot_heatmap(
    aggregate_rows: Sequence[Dict[str, Any]],
    *,
    row_field: str,
    col_field: str,
    value_field: str,
    title: str,
    output_file: Path,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (7.2, 5.0),
    dpi: int = 200,
) -> None:
    """Heatmap over two swept axes. Cells with no run are left blank (NaN)."""
    if not aggregate_rows:
        return

    rows_vals = sorted({r[row_field] for r in aggregate_rows})
    cols_vals = sorted({r[col_field] for r in aggregate_rows})
    if len(rows_vals) < 1 or len(cols_vals) < 1:
        return

    grid = np.full((len(rows_vals), len(cols_vals)), np.nan, dtype=float)
    r_index = {v: i for i, v in enumerate(rows_vals)}
    c_index = {v: i for i, v in enumerate(cols_vals)}
    for row in aggregate_rows:
        grid[r_index[row[row_field]], c_index[row[col_field]]] = float(row[value_field])

    plt.figure(figsize=figsize)
    im = plt.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
    plt.colorbar(im, label=title)
    plt.xticks(range(len(cols_vals)), [f"{v:g}" if isinstance(v, float) else str(v)
                                       for v in cols_vals])
    plt.yticks(range(len(rows_vals)), [f"{v:g}" if isinstance(v, float) else str(v)
                                       for v in rows_vals])
    plt.xlabel(col_field)
    plt.ylabel(row_field)
    plt.title(title)

    finite = np.isfinite(grid)
    if finite.any():
        lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
        mid = 0.5 * (lo + hi)
        for i in range(len(rows_vals)):
            for j in range(len(cols_vals)):
                if not np.isfinite(grid[i, j]):
                    continue
                plt.text(
                    j, i, f"{grid[i, j]:.3g}",
                    ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] < mid else "black",
                )

    _save(output_file, dpi=dpi, tight_bbox=True)


def plot_series_by_group(
    rows: Sequence[Dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    group_field: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_file: Path,
    max_legend: int = 10,
    figsize: Tuple[float, float] = (7.2, 4.6),
    dpi: int = 180,
) -> None:
    """One line per distinct `group_field` value (typically per seed)."""
    if not rows:
        return
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r[group_field], []).append(r)

    plt.figure(figsize=figsize)
    for key in sorted(groups.keys()):
        series = sorted(groups[key], key=lambda r: r[x_field])
        plt.plot(
            [r[x_field] for r in series],
            [r[y_field] for r in series],
            linewidth=1.5, alpha=0.9, label=f"{group_field} {key}",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    if len(groups) <= max_legend:
        plt.legend(fontsize=8, ncol=2)
    _save(output_file, dpi=dpi)


def plot_mean_band(
    rows: Sequence[Dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_file: Path,
    ylim: Optional[Tuple[float, float]] = None,
    hline: Optional[float] = None,
    figsize: Tuple[float, float] = (7.2, 4.6),
    dpi: int = 220,
) -> None:
    """Mean across groups at each x, with a 95% band. Used by the Experiment A figure."""
    if not rows:
        return

    buckets: Dict[Any, List[float]] = {}
    for row in rows:
        buckets.setdefault(row[x_field], []).append(float(row[y_field]))

    xs = sorted(buckets.keys())
    means = np.array([np.mean(buckets[x]) for x in xs], dtype=float)
    stds = np.array(
        [np.std(buckets[x], ddof=1) if len(buckets[x]) >= 2 else 0.0 for x in xs],
        dtype=float,
    )
    counts = np.array([len(buckets[x]) for x in xs], dtype=float)
    ci95 = 1.96 * stds / np.sqrt(np.maximum(counts, 1))

    plt.figure(figsize=figsize)
    plt.plot(xs, means, linewidth=2.5)
    if np.max(ci95) > 0:
        plt.fill_between(xs, means - ci95, means + ci95, alpha=0.2)
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(alpha=0.25)
    _save(output_file, dpi=dpi, tight_bbox=True)
