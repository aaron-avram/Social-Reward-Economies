"""
Exact parity: roleupdate.update_roles_sequential vs the benchmark's
_update_roles_sequential.

Both sides get an identical follow graph, reputation matrix, and reward estimates.

THE SHUFFLE MUST BE PINNED. Section 7 processes C in a random order (benchmark
line 2230), and the [ROLE-3] redirect makes the outcome order-DEPENDENT: when many
agents want to follow, whoever moves first becomes the root everyone else is
redirected onto. With free shuffles the two sides converge on different single
leaders — that is a draw difference, not a divergence. Pinning both to identity
order isolates the algorithm.
"""
import numpy as np
import pytest

from .harness import (
    bench_system, load_benchmark, read_bench_graph, read_pkg_graph, set_bench_rep_state,
)
from model.agent import Agent
from model.config import AlgorithmParams, Dimensions
from model.roleupdate import update_roles_sequential

bm = load_benchmark()
N = 6
DIMS = Dimensions(num_agents=N, num_states=3, num_actions=2)


class PinnedOrder:
    """An rng whose shuffle is the identity, matching the pinned benchmark."""
    def shuffle(self, x):
        return None

    def choice(self, a, **kw):
        return np.random.default_rng(0).choice(a, **kw)


@pytest.fixture(autouse=True)
def pin_benchmark_shuffle(monkeypatch):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)


def build_pair(seed, *, gamma=2.0, kappa=2.0, c_threshold=0.1):
    """Two systems in the same state: a benchmark one and a list of package Agents."""
    g = np.random.default_rng(seed)
    s = g.normal(size=(N, N))
    v = np.zeros((N, N))
    L = np.array([int(np.delete(np.arange(N), i)[np.argmax(s[i, np.delete(np.arange(N), i)])])
                  for i in range(N)])
    pu = g.normal(scale=0.3, size=N)
    st = g.normal(scale=0.3, size=N)

    sysb = bench_system(bm, N)
    sysb.config.gamma = gamma
    sysb.config.kappa = kappa
    sysb.config.c_threshold = c_threshold
    set_bench_rep_state(sysb, v, s, L)
    for i, a in enumerate(sysb.agents):
        a.state.estimated_reward_pu = float(pu[i])
        a.state.estimated_reward_status = float(st[i])

    params = AlgorithmParams(gamma=gamma, kappa=kappa, c_threshold=c_threshold)
    agents = [Agent(i, params, DIMS, np.random.default_rng(100 + i)) for i in range(N)]
    for i, a in enumerate(agents):
        a.state.estimated_reward_pu = float(pu[i])
        a.state.estimated_reward_status = float(st[i])

    return sysb, agents, params, s, L


def run_both(sysb, agents, params, s, L, times=1):
    for _ in range(times):
        sysb._update_roles_sequential()
        update_roles_sequential(agents, s, L, params, PinnedOrder())
    return read_pkg_graph(agents), read_bench_graph(sysb)


@pytest.mark.parametrize("seed", range(15))
def test_role_assignment_matches(seed):
    pkg, bench = run_both(*build_pair(seed))
    assert pkg[0] == bench[0], "role assignment differs"
    assert pkg[1] == bench[1], "follow targets differ"
    assert pkg[2] == bench[2], "follower sets differ"


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0, 5.0])
def test_matches_across_gamma(seed, gamma):
    """gamma scales the step-1 follow signal — sweeps the decision boundary."""
    pkg, bench = run_both(*build_pair(seed, gamma=gamma))
    assert pkg == bench


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("kappa", [0.0, 2.0, 10.0])
def test_matches_across_kappa(seed, kappa):
    """kappa gates step 2 — exercises the STATUS branch."""
    pkg, bench = run_both(*build_pair(seed, kappa=kappa))
    assert pkg == bench


@pytest.mark.parametrize("seed", range(10))
def test_two_consecutive_updates_match(seed):
    """The second update starts from a non-trivial follow graph, exercising
    hysteresis, the [ROLE-3] redirect, and [ROLE-5] rehoming."""
    pkg, bench = run_both(*build_pair(seed), times=2)
    assert pkg == bench


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("c", [0.0, 1 / N, 0.5, 1.0])
def test_status_threshold_boundary_matches(seed, c):
    """|F_i| >= ceil(cN) — the ceil-vs-floor distinction at the boundary."""
    pkg, bench = run_both(*build_pair(seed, c_threshold=c, kappa=10.0))
    assert pkg == bench


@pytest.mark.parametrize("seed", range(8))
def test_subset_update_matches(seed):
    """Async mode: only a subset of agents reevaluate."""
    sysb, agents, params, s, L = build_pair(seed)
    subset = [0, 2, 4]
    sysb._update_roles_sequential(update_candidates=subset)
    update_roles_sequential(agents, s, L, params, PinnedOrder(),
                            update_candidates=subset)
    assert read_pkg_graph(agents) == read_bench_graph(sysb)
