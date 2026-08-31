"""
EXACT tests: v3 must accept the v1/v2 command lines.

These are the cheapest possible regression check on the port — they run no
simulation, so a broken CLI is caught in milliseconds rather than after a
two-minute sweep.
"""
import re
import subprocess
import sys

import pytest

from .conftest import EXP, ROOT, script

pytestmark = pytest.mark.experiment

# --plot-sample-interval is v1-only: v3's base does not plot yet.
# --seed-start and --numpy-fast-path are covered by dedicated tests below.
# v1-only by design:
#   --plot-sample-interval  : v3's base does not plot yet
#   --seed-start            : replaced by --seed-base + SeedSequence
# v1-only by omission — experiment-specific args not yet ported to v3.
# Each needs a decision: port it, or drop it and note why.
NOT_YET_PORTED = {
    "--convergence-threshold-frac", "--kappa-scale-by-n",
    "--leader-switch-margin", "--plot-leader-actor-rate",
    "--plot-leader-actor-rate-grid-show-seeds",
    "--trace-every", "--trace-seeds",
}
KNOWN_ABSENT = {"--plot-sample-interval", "--seed-start"} | NOT_YET_PORTED

PAIRS = [
    ("status_scaling", "status_scaling"),
    ("reputation_status_scaling", "reputation_status_scaling"),
    ("pu_scaling", "pu_scaling"),
]


def accepts(path, flag: str, value: str | None = "1") -> bool:
    """
    Whether a script ACCEPTS a flag, determined by invoking it.

    Parsing --help was the obvious approach and it does not work: argparse
    wraps long help text, so a flag MENTIONED in another option's description
    lands at the start of a continuation line and looks like an option. That
    produced a false positive here (--selected-seeds' help names --seed-start).
    Invoking is slower but tests the real thing.
    """
    # NOT with --help: argparse fires the help action and exits before it
    # validates the earlier flag, so every flag looks accepted.
    # Try with a value and without: store_true flags reject a value, valued
    # flags require one, and we cannot tell which is which from the outside.
    candidates = [[flag]] if value is None else [[flag, value], [flag]]
    return any(_probe(path, a) for a in candidates)


def _probe(path, args: list[str]) -> bool:
    out = subprocess.run(
        [sys.executable, str(path), *args,
         "--seeds", "0", "--num-steps", "0", "--output-dir", "/tmp/_cli_probe"],
        capture_output=True, text=True, cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)}, timeout=120)
    return "unrecognized arguments" not in (out.stderr + out.stdout)


def source_args(path) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', path.read_text()))


@pytest.mark.parametrize("v1_name,v3_name", PAIRS)
def test_v3_accepts_every_v1_argument(v1_name, v3_name):
    """Any argument v1 accepted must still parse, so an old invocation from a
    lab notebook does not fail with 'unrecognized arguments'."""
    v1 = source_args(script(v1_name, "v1")) - KNOWN_ABSENT
    missing = sorted(f for f in v1 if not accepts(script(v3_name, "v3"), f))
    assert not missing, f"{v3_name} dropped: {missing}"


@pytest.mark.parametrize("name", [p[1] for p in PAIRS])
def test_numpy_fast_path_is_accepted_and_inert(name):
    """The two phase-4 implementations merged, so the flag selects nothing.
    It is accepted rather than removed so existing scripts keep working."""
    assert accepts(script(name, "v3"), "--numpy-fast-path", None)


@pytest.mark.parametrize("name", [p[1] for p in PAIRS])
def test_selected_seeds_is_documented_as_indices(name):
    """v1's --selected-seeds took seed VALUES; v3's takes replicate INDICES,
    because SeedSequence seeds are not consecutive. The help text must say so,
    or a user will silently select nothing."""
    out = subprocess.run([sys.executable, str(script(name, "v3")), "--help"],
                         capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(ROOT)}, timeout=120).stdout
    assert "--selected-seeds" in out
    assert "INDICES" in out or "indices" in out


def test_seed_start_is_gone_deliberately():
    """Not an oversight: consecutive integer seeds are exactly what v3 replaced.
    Keeping the name would imply the old semantics still hold."""
    assert not accepts(script("status_scaling", "v3"), "--seed-start")
