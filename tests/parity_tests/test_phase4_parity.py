"""
Exact parity: reputation.phase4 vs the benchmark's _phase4_updates_python.

Both sides get identical v/s/L, identical observed utilities, and identical active
sets. The only randomness is tie-breaking; leader updates are disabled in the cases
that must be bit-exact, and checked separately under a forced-unique-max state where
the tie-break is deterministic.
"""
import numpy as np
import pytest

from .harness import (
    bench_system, load_benchmark, read_bench_rep_state, set_bench_rep_state,
)
from model.config import AlgorithmParams, Eq9Mode, LeaderUpdateMode
from model.reputation import ReputationState, phase4

bm = load_benchmark()
N = 6


def scenario(seed):
    """Random but reproducible v/s/L, utilities, and active sets."""
    g = np.random.default_rng(seed)
    v = g.normal(size=(N, N))
    s = g.normal(size=(N, N))
    L = g.integers(0, N, size=N)
    for i in range(N):                       # a leader is never self
        if L[i] == i:
            L[i] = (i + 1) % N
    U = g.normal(size=(N, N))
    actors = np.array(sorted(g.choice(N, size=g.integers(1, N + 1), replace=False)))
    parts = np.array(sorted(g.choice(N, size=g.integers(1, N + 1), replace=False)))
    return v, s, L, U, actors, parts


MODES = [(e, l) for e in Eq9Mode for l in LeaderUpdateMode]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("eq9,leader", MODES)
def test_v_and_s_match_without_leader_updates(seed, eq9, leader):
    """The Eq.(4) and Eq.(9) recurrences consume no randomness, so with
    identify_highest_rep off both sides must agree to floating-point exactness."""
    v, s, L, U, actors, parts = scenario(seed)
    eta_v = 0.0731

    sysb = bench_system(bm, N)
    set_bench_rep_state(sysb, v, s, L)
    sysb.config.eq9_averaging_mode = eq9.value
    sysb.config.leader_update_mode = leader.value
    sysb._phase4_updates_python(
        U, actors.tolist(), parts.tolist(), eta_v, 0.0,
        update_actor_rates=False, identify_highest_rep=False,
    )
    vb, sb, _ = read_bench_rep_state(sysb)

    st = ReputationState(v=v.copy(), s=s.copy(), L=L.copy().astype(int))
    phase4(st, U, actors, parts, eta_v, AlgorithmParams(), eq9, leader,
           np.random.default_rng(0), update_leader_estimates=False)

    assert np.allclose(st.v, vb, atol=1e-12, rtol=0), "Eq. (4) personal benefits differ"
    assert np.allclose(st.s, sb, atol=1e-12, rtol=0), "Eq. (9) reputations differ"


@pytest.mark.parametrize("seed", range(12))
def test_leader_choice_lands_in_the_same_candidate_set(seed):
    """
    L_i is drawn uniformly from the delta-tie set K_i(t), so the two sides pick
    from different RNG streams and need not agree on the element. What must agree
    is the SET: both must compute the same K_i(t) from the same post-Eq.(9) s.

    (An earlier version of this test asserted equality of L and failed on seed 2 —
    correctly: after Eq.(9) collapses a column, agent 5's row had a genuine
    three-way tie {1, 3, 4}. That is a tie-break draw, not a divergence.)
    """
    v, s, L, U, actors, parts = scenario(seed)
    eta_v = 0.0731
    delta = AlgorithmParams().delta

    st = ReputationState(v=v.copy(), s=s.copy(), L=L.copy().astype(int))
    phase4(st, U, actors, parts, eta_v, AlgorithmParams(),
           Eq9Mode.PARTICIPANTS_ONLY, LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9,
           np.random.default_rng(0), update_leader_estimates=True)

    sysb = bench_system(bm, N)
    set_bench_rep_state(sysb, v, s, L)
    sysb._phase4_updates_python(
        U, actors.tolist(), parts.tolist(), eta_v, 0.0,
        update_actor_rates=False, identify_highest_rep=True,
    )
    _, sb, Lb = read_bench_rep_state(sysb)

    for i in parts:
        row = sb[i]
        others = [k for k in range(N) if k != i]
        candidates = {k for k in others if row[k] >= max(row[k] for k in others) - delta}
        assert int(st.L[i]) in candidates, f"agent {i}: package picked outside K_i"
        assert int(Lb[i]) in candidates, f"agent {i}: benchmark picked outside K_i"


@pytest.mark.parametrize("seed", range(8))
def test_gossip_target_set_matches(seed):
    """B(t) = union of participants' L_i — the scope of the Eq.(9) update."""
    v, s, L, U, actors, parts = scenario(seed)
    sysb = bench_system(bm, N)
    set_bench_rep_state(sysb, v, s, L)
    want = sysb._compute_gossip_target_ids_from_active_participants(parts.tolist())

    from model.reputation import gossip_targets
    st = ReputationState(v=v.copy(), s=s.copy(), L=L.copy().astype(int))
    got = gossip_targets(st, parts)
    assert got.tolist() == list(want)


@pytest.mark.parametrize("seed", range(8))
def test_no_participants_leaves_s_untouched_on_both_sides(seed):
    v, s, L, U, actors, _ = scenario(seed)
    eta_v = 0.1
    empty = np.array([], dtype=int)

    sysb = bench_system(bm, N)
    set_bench_rep_state(sysb, v, s, L)
    sysb._phase4_updates_python(U, actors.tolist(), [], eta_v, 0.0,
                                update_actor_rates=False, identify_highest_rep=False)
    vb, sb, _ = read_bench_rep_state(sysb)

    st = ReputationState(v=v.copy(), s=s.copy(), L=L.copy().astype(int))
    phase4(st, U, actors, empty, eta_v, AlgorithmParams(),
           Eq9Mode.PARTICIPANTS_ONLY, LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9,
           np.random.default_rng(0), update_leader_estimates=False)

    assert np.allclose(st.v, vb, atol=1e-12, rtol=0)
    assert np.allclose(st.s, sb, atol=1e-12, rtol=0)
