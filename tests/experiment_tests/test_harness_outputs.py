"""
End-to-end: run each v1 and v2 harness as a subprocess and compare their CSVs.

This is the only layer that exercises the harness code itself — argument parsing,
metric computation, aggregation, file naming. Everything above tests the engine
and the adapter; a harness could still be broken in a way those miss.

SCHEMA is compared exactly (columns and row count are deterministic).
VALUES are compared distributionally, for the RNG reason in conftest.

These are slow (each invocation is a full sweep), so the grids are deliberately
small. They are still the longest tests in the repo — hence the `experiment`
marker and the `--runslow`-style opt-in.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"

pytestmark = pytest.mark.experiment

# Small but ACTIVE: B_R=0.5 with a high gamma forms followers (see conftest).
#
# SEEDS OVER GRID POINTS. 12 seeds on a 2x2 grid, not 4 seeds on the default
# 5x5: the comparison's power comes from seed count, while grid points mostly
# add runtime. At 4 seeds the leader identity is effectively fixed per grid
# point, and since each agent prefers a different action under
# simple_preferred_action, a fixed leader shifts welfare by a fixed amount and
# the test reports a spurious difference. Measured: the offset shrank
# 0.0030 -> 0.0008 -> 0.00008 as seeds went 4 -> 8 -> 12.
COMMON = [
    "--num-agents", "6", "--num-steps", "300", "--seeds", "12",
    "--B-R", "0.5", "--B-F", "0.35",
]
SMALL_GRID = ["--gammas", "2,5", "--kappas", "0,2"]

HARNESSES = [
    ("pu_scaling", COMMON + ["--mode", "static", "--num-states-list", "3"]),
    ("status_scaling", COMMON + SMALL_GRID + ["--mode", "static"]),
    ("status_scaling_async", COMMON + SMALL_GRID + ["--mode", "async"]),
    ("reputation_status_scaling", COMMON + SMALL_GRID + ["--mode", "static"]),
]


def _script(name: str, version: str) -> str:
    base = name.replace("_async", "")
    return str(EXP / f"{base}_{version}.py")


def _run(name: str, version: str, args: list[str], outdir: Path) -> None:
    cmd = [sys.executable, _script(name, version), *args,
           "--output-dir", str(outdir)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"{name} {version} failed:\n{proc.stderr[-3000:]}"


def _read_csv(path: Path):
    import csv
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _runs_csv(outdir: Path) -> Path:
    hits = sorted(outdir.rglob("*_runs_*.csv"))
    assert hits, f"no per-run CSV under {outdir}"
    return hits[0]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    """Run every harness in both versions once, and reuse across tests."""
    root = tmp_path_factory.mktemp("harness")
    made = {}
    for name, args in HARNESSES:
        for version in ("v1", "v2"):
            outdir = root / f"{name}_{version}"
            _run(name, version, args, outdir)
            made[(name, version)] = _read_csv(_runs_csv(outdir))
    return made


@pytest.mark.parametrize("name", [h[0] for h in HARNESSES])
def test_both_versions_run(outputs, name):
    assert outputs[(name, "v1")] and outputs[(name, "v2")]


@pytest.mark.parametrize("name", [h[0] for h in HARNESSES])
def test_csv_columns_are_identical(outputs, name):
    """The ported harness must emit the same schema — downstream analysis
    scripts read these by column name."""
    assert list(outputs[(name, "v1")][0]) == list(outputs[(name, "v2")][0])


@pytest.mark.parametrize("name", [h[0] for h in HARNESSES])
def test_row_count_is_identical(outputs, name):
    """One row per (grid point, seed) — deterministic, not RNG-dependent."""
    assert len(outputs[(name, "v1")]) == len(outputs[(name, "v2")])


@pytest.mark.parametrize("name", [h[0] for h in HARNESSES])
def test_parameter_columns_match_exactly(outputs, name):
    """Columns echoing the config must be identical row for row — these are
    inputs, not outcomes, so any difference is a translation bug."""
    v1, v2 = outputs[(name, "v1")], outputs[(name, "v2")]
    param_cols = [c for c in v1[0]
                  if c in {"mode", "seed", "gamma", "kappa", "num_agents",
                           "num_states", "num_actions", "num_steps",
                           "reward_model", "c_threshold", "B_R", "B_F"}]
    assert param_cols, "no recognisable parameter columns"
    for col in param_cols:
        assert [r[col] for r in v1] == [r[col] for r in v2], f"{name}: {col} differs"


def _numeric_columns(rows):
    out = []
    for col in rows[0]:
        try:
            [float(r[col]) for r in rows]
        except (ValueError, TypeError):
            continue
        out.append(col)
    return out


def _group_key(row):
    """Identify a grid point: every parameter column except the seed."""
    return tuple(sorted(
        (k, v) for k, v in row.items()
        if k in {"mode", "gamma", "kappa", "num_agents", "num_states",
                 "num_actions", "num_steps", "reward_model"}))


def _grid_point_means(rows, col):
    """Mean of `col` within each grid point, keyed so the two versions align."""
    buckets = {}
    for r in rows:
        try:
            value = float(r[col])
        except (ValueError, TypeError):
            return None
        if not np.isfinite(value):
            return None
        buckets.setdefault(_group_key(r), []).append(value)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def _seed_level_se(v1, v2, col):
    """
    Standard error of the mean difference from the SEED-level spread.

    Used as a noise floor. The paired-by-grid-point test has no power when a
    metric does not vary across grid points (gamma and kappa leave tail welfare
    unchanged here, so every pair shows the identical difference and the paired
    sd is exactly zero). The seed-level spread is the real sampling noise in
    that case.
    """
    a = np.array([float(r[col]) for r in v1])
    b = np.array([float(r[col]) for r in v2])
    return np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


@pytest.mark.parametrize("name", [h[0] for h in HARNESSES])
def test_numeric_outcomes_agree_distributionally(outputs, name):
    """
    Every numeric outcome column, compared PAIRED BY GRID POINT with a
    seed-level noise floor.

    Two corrections learned the hard way, both worth keeping:

    1. Pooling all rows is wrong. Rows differ by seed (random) AND by grid point
       (systematic), so the pooled standard error understates the sampling noise.
       Measured: tail_welfare gave t=7.7 pooled at 4 seeds but t=1.6 at 20 — the
       signature of an artifact, not a bias.

    2. Pairing alone is not enough either. When a metric is constant across grid
       points, every pair shows the same difference, the paired sd is 0, and the
       t-statistic is infinite for an arbitrarily small offset. Falling back to
       the seed-level SE fixes that; measured, the offending offset shrank
       0.0030 -> 0.0008 -> 0.00008 as seeds went 4 -> 8 -> 12, converging to
       zero exactly as an RNG difference should.

    Reports ALL offending columns rather than the first, so one run tells you
    whether a failure is isolated or systemic.
    """
    v1, v2 = outputs[(name, "v1")], outputs[(name, "v2")]
    failures = []

    for col in _numeric_columns(v1):
        m1 = _grid_point_means(v1, col)
        m2 = _grid_point_means(v2, col)
        if m1 is None or m2 is None:
            continue
        shared = sorted(set(m1) & set(m2))
        if len(shared) < 2:
            continue

        diffs = np.array([m1[k] - m2[k] for k in shared])
        if np.abs(diffs).max() < 1e-9:
            continue

        paired_se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        se = max(paired_se, _seed_level_se(v1, v2, col))
        if se == 0:
            failures.append(
                f"{col}: offset {diffs.mean():+.4g} with zero variance on both sides")
            continue

        t = abs(diffs.mean()) / se
        if t >= 4.0:
            failures.append(
                f"{col}: mean paired diff {diffs.mean():+.4g} "
                f"(t={t:.2f}, {len(shared)} grid points)")

    assert not failures, f"{name}:\n  " + "\n  ".join(failures)
