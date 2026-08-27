"""
Experiment-level comparison: the ported harnesses (v2, new engine) against the
originals (v1, frozen engine).

WHY THESE ARE DISTRIBUTIONAL, NOT EXACT
---------------------------------------
The old engine drew everything from the global stream, so np.random.seed(s) in
run_single() fixed the trajectory. The new engine uses config.runtime.seed and
five spawned substreams, and compat.py derives that seed from the global stream.
No seed pair makes the two consume draws in the same order, so a given seed is
NOT the same run on both sides — it is a different draw from the same
distribution. Comparisons are therefore over seed ensembles.

Exact assertions are still possible, and used below, for anything downstream of
a FIXED input: CSV schema, config translation, metric functions.

ACTIVE REGIME
-------------
gamma=5.0, B_R=0.5, shared_good_bad_heterogeneous, 400 steps, N=6 produces real
follower formation (max followers 4-5 across seeds). The defaults do NOT — at
N=5/100 steps nothing follows, every role metric is identically zero, and the
comparison passes vacuously. Do not lower these without re-checking that
test_active_regime_actually_forms_followers still passes.
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.experiment

ROOT = Path(__file__).resolve().parents[2]

# Active-regime parameters, shared by every comparison in this directory.
ACTIVE = dict(
    num_agents=6, num_states=3, num_actions=2, num_time_steps=400,
    gamma=5.0, kappa=2.0, B_R=0.5, B_F=0.35,
    role_update_base_interval=25,
    reward_model="shared_good_bad_heterogeneous",
)
N_SEEDS = 12


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def old_engine():
    """The frozen pre-refactor engine, as the v1 harnesses import it."""
    return _load(ROOT / "src" / "benchmark_code.py", "_old_engine")


@pytest.fixture(scope="session")
def new_engine():
    """compat.SystemConfig / MultiAgentSystem — the v2 harnesses' entry point."""
    from experiments.v2 import compat
    return compat


def run_ensemble(engine, *, n_seeds=N_SEEDS, **overrides):
    """
    Run `n_seeds` simulations and return one summary dict per seed.

    Both engines are driven identically: np.random.seed(s), build config from the
    same flat kwargs, step num_time_steps times. Everything that differs between
    them is inside the engine, which is what the comparison is measuring.
    """
    params = {**ACTIVE, **overrides}
    steps = int(params["num_time_steps"])
    out = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        system = engine.MultiAgentSystem(engine.SystemConfig(**params))
        with redirect_stdout(io.StringIO()):
            for _ in range(steps):
                system.step()
        out.append(summarise(system))
    return out


def summarise(system) -> dict:
    """The outcome statistics the four harnesses actually report on."""
    agents = system.agents
    followers = [len(a.state.followers) for a in agents]
    roles = [a.state.role.value for a in agents]
    results = system.results
    welfare = results["paper_welfare_followers_only"]
    return {
        "max_followers": max(followers),
        "n_reputation": roles.count("reputation"),
        "n_status": roles.count("status"),
        "n_personal_utility": roles.count("personal_utility"),
        "has_leader": int(max(followers) > 0),
        "final_welfare": float(welfare[-1]),
        "mean_welfare": float(np.mean(welfare)),
        "mean_actor_rate": float(np.mean([a.state.actor_interaction_rate for a in agents])),
        "n_role_updates": len(results["role_update_times"]),
    }


def welch_t(a, b) -> float:
    """|t| for the difference of means. Returns 0.0 when both are degenerate and
    equal, inf when degenerate and unequal."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = abs(a.mean() - b.mean())
    if diff < 1e-9:
        return 0.0
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float("inf") if se == 0 else diff / se
