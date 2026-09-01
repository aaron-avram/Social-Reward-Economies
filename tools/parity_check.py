#!/usr/bin/env python3
"""
Differential test: v2 harness versus v3 port, same arguments, same CSVs.

This is the acceptance criterion for the refactor. Each case runs both harnesses
into separate temp directories and compares every CSV cell-by-cell, with a
tolerance for float formatting.

Usage:
    PYTHONPATH=src python3 tools/parity_check.py            # quick cases
    PYTHONPATH=src python3 tools/parity_check.py --full     # includes async
    PYTHONPATH=src python3 tools/parity_check.py -k exp_a   # one case

Note the default --seed-derivation is legacy_global, which is what makes this
comparison possible at all: the v2 outputs were produced through the compat
shim, which derived the engine seed from the global RNG rather than passing the
run seed to the engine. See runner.resolve_engine_seed.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FLOAT_TOL = 1e-12


@dataclass
class Case:
    name: str
    legacy: str
    ported: str
    args: List[str]
    #: CSV stems to compare; empty means "every CSV both sides produced".
    only: Sequence[str] = field(default_factory=tuple)
    slow: bool = False


CASES: List[Case] = [
    Case(
        name="exp_a_static",
        legacy="src/experiments/v2/pu_scaling.py",
        ported="src/experiments/v3/exp_a_pu_scaling.py",
        args=[
            "--mode", "static", "--num-agents", "8", "--num-states-list", "3,4",
            "--num-steps", "400", "--seeds", "3",
            "--role-update-base-interval", "100",
            "--trace-seeds", "0", "--trace-every", "50",
        ],
    ),
    Case(
        name="exp_a_async",
        legacy="src/experiments/v2/pu_scaling.py",
        ported="src/experiments/v3/exp_a_pu_scaling.py",
        args=[
            "--mode", "async", "--num-agents", "8", "--num-states-list", "3",
            "--num-steps", "400", "--seeds", "3",
            "--role-update-base-interval", "80",
        ],
    ),
    Case(
        name="exp_c_static",
        legacy="src/experiments/v2/status_scaling.py",
        ported="src/experiments/v3/exp_c_status_scaling.py",
        args=[
            "--mode", "static", "--num-agents", "8", "--num-states", "3",
            "--num-steps", "400", "--seeds", "3",
            "--gammas", "0,2", "--kappas", "0,1",
            "--role-update-base-interval", "100",
        ],
    ),
    Case(
        name="exp_b_static",
        legacy="src/experiments/v2/reputation_status_scaling.py",
        ported="src/experiments/v3/exp_b_reputation_status_scaling.py",
        args=[
            "--mode", "static", "--num-agents", "8", "--num-states", "3",
            "--num-steps", "400", "--seeds", "3",
            "--gammas", "0,2", "--kappas", "0,1",
            "--role-update-base-interval", "100",
        ],
    ),
    Case(
        name="exp_b_async",
        legacy="src/experiments/v2/reputation_status_scaling.py",
        ported="src/experiments/v3/exp_b_reputation_status_scaling.py",
        args=[
            "--mode", "async", "--num-agents", "8", "--num-states", "3",
            "--num-steps", "400", "--seeds", "2",
            "--gammas", "2", "--kappas", "0,1",
            "--role-update-base-interval", "80",
        ],
        slow=True,
    ),
    Case(
        name="exp_d_static",
        legacy="src/experiments/v2/perturbation_recovery.py",
        ported="src/experiments/v3/exp_d_perturbation_recovery.py",
        args=[
            "--mode", "static", "--num-agents", "8", "--num-states", "3",
            "--num-steps-max", "1200", "--seeds", "2",
            "--gamma", "5", "--kappa", "0",
            "--role-update-base-interval", "60",
            "--conv-threshold", "0.5", "--conv-hold-steps", "5",
            "--recovery-threshold", "0.4", "--recovery-hold-steps", "5",
            "--perturb-duration", "50", "--post-window", "100",
            "--no-auto-run-subdir",
        ],
        slow=True,
    ),
    Case(
        name="exp_d_async",
        legacy="src/experiments/v2/perturbation_recovery.py",
        ported="src/experiments/v3/exp_d_perturbation_recovery.py",
        args=[
            "--mode", "async", "--num-agents", "8", "--num-states", "3",
            "--num-steps-max", "1200", "--seeds", "2",
            "--gamma", "5", "--kappa", "0",
            "--role-update-base-interval", "60",
            "--conv-threshold", "0.5", "--conv-hold-steps", "5",
            "--recovery-threshold", "0.4", "--recovery-hold-steps", "5",
            "--perturb-duration", "50", "--post-window", "100",
            "--no-auto-run-subdir",
        ],
        slow=True,
    ),
]


def run(script: str, args: Sequence[str], out_dir: Path) -> Tuple[int, str]:
    cmd = [sys.executable, str(ROOT / script), *args, "--output-dir", str(out_dir)]
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin:/usr/local/bin",
           "MPLBACKEND": "Agg", "HOME": str(Path.home())}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    return proc.returncode, (proc.stdout + proc.stderr)


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def cells_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa != fa and fb != fb:  # both NaN
        return True
    return abs(fa - fb) <= FLOAT_TOL * max(1.0, abs(fa), abs(fb))


def compare_dir(legacy_dir: Path, ported_dir: Path,
                only: Sequence[str]) -> List[str]:
    problems: List[str] = []
    legacy_csvs = {p.name: p for p in sorted(legacy_dir.rglob("*.csv"))}
    ported_csvs = {p.name: p for p in sorted(ported_dir.rglob("*.csv"))}

    names = sorted(set(legacy_csvs) | set(ported_csvs))
    if only:
        names = [n for n in names if any(o in n for o in only)]
    if not names:
        problems.append("no CSVs produced on either side")
        return problems

    for name in names:
        if name not in legacy_csvs:
            problems.append(f"{name}: only the port produced it")
            continue
        if name not in ported_csvs:
            problems.append(f"{name}: only the legacy harness produced it")
            continue

        lhead, lrows = read_csv(legacy_csvs[name])
        phead, prows = read_csv(ported_csvs[name])

        if lhead != phead:
            problems.append(
                f"{name}: header mismatch\n"
                f"    legacy: {lhead}\n"
                f"    ported: {phead}"
            )
            continue
        if len(lrows) != len(prows):
            problems.append(
                f"{name}: {len(lrows)} legacy rows vs {len(prows)} ported rows"
            )
            continue

        for i, (lr, pr) in enumerate(zip(lrows, prows)):
            for col in lhead:
                if not cells_equal(lr[col], pr[col]):
                    problems.append(
                        f"{name}: row {i} column {col!r}: "
                        f"legacy={lr[col]!r} ported={pr[col]!r}"
                    )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="include slow cases")
    ap.add_argument("-k", dest="pattern", default="", help="substring filter on case name")
    ap.add_argument("--keep", action="store_true", help="keep temp output dirs")
    opts = ap.parse_args()

    cases = [c for c in CASES if opts.full or not c.slow]
    if opts.pattern:
        cases = [c for c in cases if opts.pattern in c.name]
    if not cases:
        print("no cases selected")
        return 1

    failures = 0
    for case in cases:
        tmp = Path(tempfile.mkdtemp(prefix=f"parity_{case.name}_"))
        legacy_dir, ported_dir = tmp / "legacy", tmp / "ported"
        legacy_dir.mkdir(parents=True)
        ported_dir.mkdir(parents=True)

        print(f"=== {case.name} ".ljust(72, "="))
        rc_l, log_l = run(case.legacy, case.args, legacy_dir)
        rc_p, log_p = run(case.ported, case.args, ported_dir)

        if rc_l != 0:
            print(f"  LEGACY FAILED (rc={rc_l})\n{log_l[-2500:]}")
            failures += 1
        elif rc_p != 0:
            print(f"  PORT FAILED (rc={rc_p})\n{log_p[-2500:]}")
            failures += 1
        else:
            problems = compare_dir(legacy_dir, ported_dir, case.only)
            if problems:
                failures += 1
                print(f"  MISMATCH ({len(problems)} problem(s)):")
                for p in problems[:20]:
                    print(f"    {p}")
                if len(problems) > 20:
                    print(f"    ... and {len(problems) - 20} more")
            else:
                n = len(list(ported_dir.rglob("*.csv")))
                print(f"  OK -- {n} CSV file(s) identical")

        if opts.keep:
            print(f"  outputs kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"{len(cases) - failures}/{len(cases)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
