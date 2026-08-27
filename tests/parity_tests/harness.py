"""Shared plumbing: load the benchmark, build matched states on both sides."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "src" / "benchmark_code.py"

def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_code", BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench_system(bm, n=6, *, seed=0, **cfg):
    """A benchmark system with its global RNG pinned, so construction is repeatable."""
    np.random.seed(seed)
    return bm.MultiAgentSystem(bm.SystemConfig(num_agents=n, **cfg))


def set_bench_rep_state(sys_, v, s, L):
    """Write v/s/L into the benchmark's per-agent dicts (and dense caches)."""
    n = sys_.config.num_agents
    for i, agent in enumerate(sys_.agents):
        agent.state.personal_benefit_estimates = {k: float(v[i, k]) for k in range(n)}
        agent.state.reputation_estimates = {k: float(s[i, k]) for k in range(n)}
        agent.state.highest_rep_agent_estimate = None if L[i] < 0 else int(L[i])
    if sys_._v_matrix is not None:
        sys_._v_matrix = np.array(v, dtype=float, copy=True)
        sys_._s_matrix = np.array(s, dtype=float, copy=True)


def read_bench_rep_state(sys_):
    n = sys_.config.num_agents
    v = np.zeros((n, n)); s = np.zeros((n, n)); L = np.full(n, -1, dtype=int)
    for i, agent in enumerate(sys_.agents):
        for k in range(n):
            v[i, k] = agent.state.personal_benefit_estimates.get(k, 0.0)
            s[i, k] = agent.state.reputation_estimates.get(k, 0.0)
        est = agent.state.highest_rep_agent_estimate
        L[i] = -1 if est is None else int(est)
    return v, s, L


def read_bench_graph(sys_):
    return (
        [a.state.role.value for a in sys_.agents],
        [(-1 if a.state.following is None else int(a.state.following)) for a in sys_.agents],
        [set(a.state.followers) for a in sys_.agents],
    )


def read_pkg_graph(agents):
    return (
        [a.state.role.value for a in agents],
        [(-1 if a.state.following is None else int(a.state.following)) for a in agents],
        [set(a.state.followers) for a in agents],
    )
