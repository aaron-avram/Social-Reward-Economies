"""
Parity-suite fixtures. Compares the package against the frozen benchmark.

Retired by deleting benchmark_code.py — this file then skips the whole
directory rather than erroring.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "src" / "benchmark_code.py"

collect_ignore_glob = ["test_*.py"] if not BENCH_PATH.exists() else []

@pytest.fixture(scope="session")
def benchmark():
    """Loaded once per session — it's a 2,700-line module and importing it
    per test is most of the suite's runtime."""
    spec = importlib.util.spec_from_file_location("benchmark_code", BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pinned_shuffle(monkeypatch):
    """
    Section 7 is ORDER-DEPENDENT: the [ROLE-3] redirect makes whoever moves
    first the root everyone else is redirected onto. Free shuffles let the two
    sides converge on different leaders — a draw difference, not a divergence.
    Tests that compare role assignment must request this.
    """
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)


SOFTMAX_EPS_TOL = 1e-6
"""The benchmark divides by (sum(exp) + 1e-8), so its policies sum to
0.999999993. Measured effect: 5.8e-9 on expected utilities, 1.1e-8 on welfare.
Parity assertions use this tolerance; test_welfare_parity pins the magnitude
separately so a real divergence can't hide inside it."""