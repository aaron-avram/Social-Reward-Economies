"""reputation.py — Eq. (4) and Eq. (9) as pure functions over arrays."""
import numpy as np
import pytest

from model.config import AlgorithmParams, Eq9Mode, LeaderUpdateMode
from model import reputation as REP
from model.reputation import NO_LEADER, ReputationState


def _state(n=5):
    return ReputationState.initial(n)


# ------------------------------------------------------- leader selection

def test_leader_selection_picks_the_maximum():
    row = np.array([0.1, 0.9, 0.2, 0.3])
    assert REP.select_leader_from_row(row, 0, 0.0, np.random.default_rng(0)) == 1


def test_leader_selection_excludes_self():
    row = np.array([10.0, 0.1, 0.2])
    for _ in range(20):
        assert REP.select_leader_from_row(row, 0, 0.05, np.random.default_rng(0)) != 0


def test_leader_selection_breaks_near_ties_within_delta():
    row = np.array([0.0, 1.00, 0.99, 0.10])
    seen = {REP.select_leader_from_row(row, 0, 0.05, np.random.default_rng(s))
            for s in range(60)}
    assert seen == {1, 2}


def test_leader_selection_returns_python_int():
    row = np.array([0.0, 1.0, 0.5])
    assert type(REP.select_leader_from_row(row, 0, 0.1, np.random.default_rng(0))) is int


def test_leader_selection_single_agent_returns_self():
    assert REP.select_leader_from_row(np.array([0.0]), 0, 0.1, np.random.default_rng(0)) == 0


def test_resolve_missing_leaders_is_idempotent_and_draws_once_per_agent():
    """A second call must be a no-op AND consume no draws, or the tiebreak stream
    desynchronises from the baseline on every subsequent step."""
    st = _state(5)
    g = np.random.default_rng(0)
    REP.resolve_missing_leaders(st, range(5), 0.1, g)
    before, state_after_first = st.L.copy(), g.bit_generator.state
    REP.resolve_missing_leaders(st, range(5), 0.1, g)
    assert np.array_equal(st.L, before)
    assert g.bit_generator.state == state_after_first


def test_resolve_missing_leaders_ignores_out_of_range():
    st = _state(3)
    REP.resolve_missing_leaders(st, [-1, 99], 0.1, np.random.default_rng(0))
    assert (st.L == NO_LEADER).all()


def test_update_leaders_overwrites_unconditionally():
    st = _state(4)
    st.L[:] = 0
    st.s[1, 3] = 5.0
    REP.update_leaders(st, [1], 0.0, np.random.default_rng(0))
    assert st.L[1] == 3


def test_update_leaders_reads_the_snapshot_when_given_one():
    st = _state(4)
    snapshot = st.s.copy()
    snapshot[1, 2] = 9.0
    st.s[1, 3] = 100.0          # live matrix says 3, snapshot says 2
    REP.update_leaders(st, [1], 0.0, np.random.default_rng(0), source_s=snapshot)
    assert st.L[1] == 2


# ------------------------------------------------------------- gossip B(t)

def test_gossip_targets_are_sorted_unique_and_exclude_self():
    st = _state(5)
    st.L[:] = [2, 2, 2, 4, NO_LEADER]     # agent 2 targets itself
    out = REP.gossip_targets(st, np.array([0, 1, 2, 3, 4]))
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [2, 4]


def test_gossip_targets_returns_int_array_when_empty():
    """np.ix_ rejects a float64 empty array."""
    st = _state(3)
    out = REP.gossip_targets(st, np.array([0, 1, 2]))
    assert out.dtype.kind == "i"
    assert out.size == 0


def test_gossip_targets_does_not_mutate():
    st = _state(4)
    st.L[:] = [1, 2, 3, 0]
    before = st.L.copy()
    REP.gossip_targets(st, np.array([0, 1]))
    assert np.array_equal(st.L, before)


# --------------------------------------------------------------- Eq. (9)

def test_eq9_averaging_ids_by_mode():
    p = np.array([1, 3])
    assert list(REP.eq9_averaging_ids(p, 5, Eq9Mode.PARTICIPANTS_ONLY)) == [1, 3]
    assert list(REP.eq9_averaging_ids(p, 5, Eq9Mode.ALL_AGENTS)) == [0, 1, 2, 3, 4]


def test_leader_update_ids_by_mode():
    p = np.array([1, 3])
    assert list(REP.leader_update_ids(p, 5, LeaderUpdateMode.ALL_AGENTS_POST_EQ9)) == [0, 1, 2, 3, 4]
    assert list(REP.leader_update_ids(p, 5, LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9)) == [1, 3]
    assert list(REP.leader_update_ids(p, 5, LeaderUpdateMode.PARTICIPANTS_ONLY_PRE_EQ9)) == [1, 3]


def test_apply_eq9_averages_then_adds_delta():
    n = 4
    s = np.zeros((n, n))
    s[:, 2] = [1.0, 2.0, 3.0, 4.0]
    dv = np.zeros((n, n))
    dv[:, 2] = 0.5
    parts = np.array([0, 1])
    s2, _ = REP.apply_eq9(s, dv, parts, np.array([0, 1, 2, 3]), np.array([2]))
    assert s2[0, 2] == pytest.approx(2.5 + 0.5)
    assert s2[1, 2] == pytest.approx(2.5 + 0.5)
    assert s2[2, 2] == pytest.approx(3.0), "non-participants untouched"


def test_apply_eq9_leaves_columns_outside_B_untouched():
    n = 4
    s = np.arange(n * n, dtype=float).reshape(n, n)
    s2, _ = REP.apply_eq9(s, np.zeros((n, n)), np.array([0, 1]),
                          np.array([0, 1]), np.array([2]))
    assert np.array_equal(s2[:, [0, 1, 3]], s[:, [0, 1, 3]])


def test_apply_eq9_does_not_mutate_input():
    s = np.ones((3, 3))
    REP.apply_eq9(s, np.ones((3, 3)), np.array([0]), np.array([0, 1, 2]), np.array([1]))
    assert np.allclose(s, 1.0)


def test_apply_eq9_empty_targets_is_a_noop():
    s = np.ones((3, 3))
    s2, avg = REP.apply_eq9(s, np.ones((3, 3)), np.array([0]), np.array([0]),
                            np.array([], dtype=int))
    assert np.allclose(s2, s) and avg == {}


def test_apply_eq9_trace_dict_only_when_requested():
    s = np.zeros((3, 3))
    _, off = REP.apply_eq9(s, np.zeros((3, 3)), np.array([0]), np.array([0, 1, 2]),
                           np.array([1]), trace=False)
    _, on = REP.apply_eq9(s, np.zeros((3, 3)), np.array([0]), np.array([0, 1, 2]),
                          np.array([1]), trace=True)
    assert off == {} and set(on) == {1}


# ------------------------------------------------------------- phase4

def _phase4(st, **kw):
    n = st.num_agents
    args = dict(
        observed_utility=np.ones((n, n)),
        active_actor_ids=np.arange(n),
        active_participant_ids=np.arange(n),
        eta_v=0.1,
        params=AlgorithmParams(),
        eq9_mode=Eq9Mode.PARTICIPANTS_ONLY,
        leader_mode=LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9,
        rng=np.random.default_rng(0),
    )
    args.update(kw)
    return REP.phase4(st, **args)


def test_phase4_returns_state_and_no_trace_by_default():
    st, trace = _phase4(_state(4))
    assert isinstance(st, ReputationState) and trace is None


def test_phase4_resolves_leaders_for_participants():
    st, _ = _phase4(_state(4))
    assert (st.L != NO_LEADER).all()


def test_phase4_updates_v():
    st = _state(4)
    _phase4(st)
    assert not np.allclose(st.v, 0.0)


def test_phase4_with_no_participants_still_updates_v():
    st = _state(4)
    _phase4(st, active_participant_ids=np.array([], dtype=int))
    assert not np.allclose(st.v, 0.0)
    assert (st.L == NO_LEADER).all()


def test_phase4_trace_is_populated_when_asked():
    st, trace = _phase4(_state(4), trace=True)
    assert trace is not None
    assert trace.delta_v is not None and trace.delta_v.shape == (4, 4)
    assert all(isinstance(x, int) for x in trace.gossip_target_ids)


def test_phase4_is_deterministic_for_a_seed():
    a, _ = _phase4(_state(5), rng=np.random.default_rng(3))
    b, _ = _phase4(_state(5), rng=np.random.default_rng(3))
    assert np.allclose(a.s, b.s) and np.array_equal(a.L, b.L)


@pytest.mark.parametrize("leader_mode", list(LeaderUpdateMode))
@pytest.mark.parametrize("eq9_mode", list(Eq9Mode))
def test_phase4_runs_under_every_mode_combination(eq9_mode, leader_mode):
    st, _ = _phase4(_state(5), eq9_mode=eq9_mode, leader_mode=leader_mode)
    assert np.isfinite(st.s).all() and np.isfinite(st.v).all()


def test_pre_eq9_mode_reads_the_snapshot_not_the_updated_matrix():
    """The audit-only mode exists precisely to select leaders from pre-Eq.(9) s.
    If the snapshot is ignored the mode is silently identical to POST_EQ9."""
    n = 4
    st = _state(n)
    st.s[:, 1] = 1.0                        # everyone rates agent 1 highest
    st.L[:] = 1
    U = np.zeros((n, n))
    U[:, 3] = 100.0                         # a huge shock toward agent 3
    pre, _ = REP.phase4(_state(n), U, np.arange(n), np.arange(n), 0.5,
                        AlgorithmParams(), Eq9Mode.PARTICIPANTS_ONLY,
                        LeaderUpdateMode.PARTICIPANTS_ONLY_PRE_EQ9,
                        np.random.default_rng(0))
    post, _ = REP.phase4(_state(n), U, np.arange(n), np.arange(n), 0.5,
                         AlgorithmParams(), Eq9Mode.PARTICIPANTS_ONLY,
                         LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9,
                         np.random.default_rng(0))
    assert np.allclose(pre.s, post.s), "the s update itself must not depend on the mode"
