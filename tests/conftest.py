"""
v3 (unified harness) against v2 (ported originals) and v1 (frozen originals).

WHAT CAN AND CANNOT BE COMPARED
-------------------------------
v1 -> v2 changed the ENGINE (different RNG streams), so those comparisons are
distributional. v2 -> v3 changes the HARNESS on the same engine — but the
comparison is still distributional, for a different reason:

  v1/v2 seeded with np.random.seed(seed_start + i), consecutive ints.
  v3 uses SeedSequence(seed_base).spawn(n), which is deliberately NOT
  consecutive: the same seed set runs at every grid point, making
  cross-grid-point comparisons paired.

That is a behaviour improvement, not a regression, so v3 cannot reproduce v2
run-for-run. It must reproduce v2's DISTRIBUTIONS.

Exact comparisons are still available and used, for everything downstream of a
fixed input: CLI surface, config construction, grid shape, CSV schema.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.experiment

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"

# Active regime: gamma=5 / B_R=0.5 forms followers. The defaults do not, and a
# dead regime makes every role metric zero on all three versions, so the whole
# comparison passes without testing anything.
ACTIVE_CLI = [
    "--num-agents", "6", "--num-steps", "300", "--seeds", "12",
    "--B-R", "0.5", "--B-F", "0.35",
    "--reward-model", "shared_good_bad_heterogeneous",
]
SMALL_GRID = ["--gammas", "2,5", "--kappas", "0,2"]


def script(name: str, version: str) -> Path:
    return EXP / version / f"{name}.py"


def run_script(name: str, version: str, args: list[str], outdir: Path) -> None:
    cmd = [sys.executable, str(script(name, version)), *args,
           "--output-dir", str(outdir)]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=1800, env=env)
    assert proc.returncode == 0, (
        f"{name} {version} failed:\n{proc.stderr[-3000:]}")


def read_runs_csv(outdir: Path) -> list[dict]:
    import csv
    hits = sorted(Path(outdir).rglob("*_runs_*.csv"))
    assert hits, f"no per-run CSV under {outdir}"
    with hits[0].open() as fh:
        return list(csv.DictReader(fh))


def numeric_columns(rows: list[dict]) -> list[str]:
    out = []
    for col in rows[0]:
        try:
            [float(r[col]) for r in rows]
        except (ValueError, TypeError):
            continue
        out.append(col)
    return out


def compare_distributions(a_rows, b_rows, *, group_by, label, skip=()):
    """
    Paired-by-grid-point comparison with a seed-level noise floor.

    Returns a list of failure strings; empty means agreement. Two corrections
    are baked in, both learned from the v1/v2 suite:

      1. Pooling all rows understates the sampling error, because rows vary by
         seed (random) AND grid point (systematic). Measured: a metric gave
         t=7.7 pooled at 4 seeds but t=1.6 at 20.
      2. Pairing alone has no power when a metric is constant across grid
         points — the paired sd is zero and t is infinite for any offset. The
         seed-level spread is the real noise in that case.
    """
    failures = []
    shared_cols = set(numeric_columns(a_rows)) & set(numeric_columns(b_rows))

    for col in sorted(shared_cols - set(skip) - set(group_by) - {"seed"}):
        ga, gb = {}, {}
        for rows, bucket in ((a_rows, ga), (b_rows, gb)):
            for r in rows:
                key = tuple(r[k] for k in group_by)
                bucket.setdefault(key, []).append(float(r[col]))
        keys = sorted(set(ga) & set(gb))
        if len(keys) < 2:
            continue

        diffs = np.array([np.mean(ga[k]) - np.mean(gb[k]) for k in keys])
        if not np.isfinite(diffs).all() or np.abs(diffs).max() < 1e-9:
            continue

        paired_se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        flat_a = np.array([v for k in keys for v in ga[k]])
        flat_b = np.array([v for k in keys for v in gb[k]])
        seed_se = np.sqrt(flat_a.var(ddof=1) / len(flat_a)
                          + flat_b.var(ddof=1) / len(flat_b))
        se = max(paired_se, seed_se)
        if se == 0:
            failures.append(f"{col}: offset {diffs.mean():+.4g}, zero variance")
            continue
        t = abs(diffs.mean()) / se
        if t >= 4.0:
            failures.append(f"{col}: mean paired diff {diffs.mean():+.4g} "
                            f"(t={t:.2f}, {len(keys)} grid points)")
    return failures
