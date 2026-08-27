"""instrumentation.py — the recorder protocol and the audit row builders."""
import json

import numpy as np
import pytest

from model.agent import Agent, AgentRole
from model.config import AlgorithmParams, Dimensions, Eq9Mode, LeaderUpdateMode
from model import instrumentation as I
from model.roleupdate import AgentSignals, RoleUpdateState


DIMS = Dimensions(num_agents=3, num_states=3, num_actions=2)


def agents(n=3):
    return [Agent(i, AlgorithmParams(),
                  Dimensions(num_agents=n, num_states=3, num_actions=2),
                  np.random.default_rng(0)) for i in range(n)]


def state(n=3):
    return RoleUpdateState(P=set(range(n)), R=set(), S=set(),
                           followers={i: set() for i in range(n)},
                           following={i: None for i in range(n)})


def signals(n=3):
    return {i: AgentSignals(role=AgentRole.PERSONAL_UTILITY, target=(i + 1) % n,
                            target_rep=0.5, estimated_reward_pu=0.1,
                            estimated_reward_status=0.2,
                            rep_row=np.zeros(n))
            for i in range(n)}


def test_null_recorder_accepts_every_protocol_call():
    r = I.NullRecorder()
    r.role_update_begin(1, state(), signals(), {0}, AlgorithmParams())
    r.role_update_step1(0, foo=1)
    r.role_update_decision(0, "X")
    r.role_update_end(state(), agents())
    r.phase4(1, None)
    r.rate_terms(1, 0, {})


def test_null_recorder_wants_nothing_expensive():
    r = I.NullRecorder()
    assert r.wants_phase4_trace is False
    assert r.wants_dense_history is False
    assert r.wants_compact_history is False


def test_full_recorder_exposes_the_same_wants_flags():
    r = I.FullRecorder()
    for name in ("wants_phase4_trace", "wants_dense_history", "wants_compact_histories"):
        assert isinstance(getattr(r, name), bool), name


def test_dense_history_implies_compact_histories():
    """enable_small_n_trace_export set BOTH flags (808-809)."""
    assert I.FullRecorder(dense_history=True).wants_compact_histories is True


def test_role_update_begin_seeds_one_row_per_updatable_agent():
    r = I.FullRecorder()
    r.role_update_begin(5, state(), signals(), {0, 2}, AlgorithmParams())
    assert set(r._pending) == {0, 2}
    assert r._pending[0]["decision_code"] == I.NOT_IN_C
    assert r._pending[0]["t"] == 5


def test_decision_is_first_write_wins():
    """Line 2368: step 3's FALLBACK_TO_PU must not clobber a step-1 decision."""
    r = I.FullRecorder()
    r.role_update_begin(1, state(), signals(), {0, 1}, AlgorithmParams())
    r.role_update_decision(0, "STAY_PU_REP_BELOW_THRESHOLD")
    r.role_update_decision(0, "FALLBACK_TO_PU")
    r.role_update_decision(1, "FALLBACK_TO_PU")
    assert r._pending[0]["decision_code"] == "STAY_PU_REP_BELOW_THRESHOLD"
    assert r._pending[1]["decision_code"] == "FALLBACK_TO_PU"


def test_decision_for_an_unscheduled_agent_is_ignored():
    r = I.FullRecorder()
    r.role_update_begin(1, state(), signals(), {0}, AlgorithmParams())
    r.role_update_decision(99, "X")


def test_step1_fields_overwrite():
    r = I.FullRecorder()
    r.role_update_begin(1, state(), signals(), {0}, AlgorithmParams())
    r.role_update_step1(0, effective_threshold=0.8, hysteresis_active=True)
    assert r._pending[0]["effective_threshold"] == 0.8
    assert r._pending[0]["hysteresis_active"] is True


def test_role_update_end_flushes_with_final_role():
    ags = agents()
    ags[0].state.role = AgentRole.REPUTATION
    ags[0].state.following = 1
    r = I.FullRecorder()
    r.role_update_begin(1, state(), signals(), {0}, AlgorithmParams())
    r.role_update_end(state(), ags)
    assert len(r.rows) == 1
    assert r.rows[0]["new_role"] == "reputation"
    assert r.rows[0]["following_after"] == 1
    assert r._pending == {}


def _tr(n=3):
    from model.welfare import TrueReputation
    return TrueReputation(
        expected_utilities=np.zeros((n, n)), theta_mu=np.ones(n),
        sum_expected_utility_others=np.ones(n), true_reputation=np.arange(n, dtype=float),
        true_rank=np.arange(1, n + 1), top_value=float(n - 1),
        exact_top_mask=np.array([False] * (n - 1) + [True]),
        near_top_mask=np.array([False] * (n - 1) + [True]),
        unique_true_top_agent=n - 1,
    )


def test_checkpoint_rows_are_json_serialisable():
    """Rows are exported as JSON/CSV: no np scalars, no raw Enums."""
    ags = agents()
    rows = I.checkpoint_bundle(
        1, ags, np.zeros((3, 3)), np.zeros(3, dtype=int), _tr(), AlgorithmParams(),
        checkpoint_kind="role_update", role_update_index=0,
        eq9_mode=Eq9Mode.PARTICIPANTS_ONLY,
        leader_mode=LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9)
    for key, group in rows.items():
        json.dumps(group)


def test_mode_columns_carry_the_enum_value_not_its_repr():
    rows = I.true_reputation_rows(
        1, _tr(), 3, checkpoint_kind="k", role_update_index=0,
        eq9_mode=Eq9Mode.PARTICIPANTS_ONLY,
        leader_mode=LeaderUpdateMode.ALL_AGENTS_POST_EQ9)
    assert rows[0]["eq9_averaging_mode"] == "participants_only"
    assert rows[0]["leader_update_mode"] == "all_agents_post_eq9"


def test_true_reputation_rows_one_per_agent():
    rows = I.true_reputation_rows(1, _tr(), 3, checkpoint_kind="k", role_update_index=0,
                                  eq9_mode=Eq9Mode.PARTICIPANTS_ONLY,
                                  leader_mode=LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9)
    assert [r["agent_id"] for r in rows] == [0, 1, 2]
    assert rows[0]["gap_to_true_top"] == pytest.approx(2.0)


def test_rate_audit_rows_compare_paper_and_code_drivers():
    ags = agents()
    rows = I.rate_audit_rows(1, ags, checkpoint_kind="k", role_update_index=0,
                             eq9_mode=Eq9Mode.PARTICIPANTS_ONLY,
                             leader_mode=LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9)
    assert all(r["paper_driver_matches_code"] == 1 for r in rows)


def test_role_update_diagnostic_row_counts_roles():
    ags = agents(4)
    ags[0].state.role = AgentRole.REPUTATION
    ags[1].state.role = AgentRole.STATUS
    row = I.role_update_diagnostic_row(7, ags, np.zeros((4, 4)), np.zeros(4, dtype=int),
                                       AlgorithmParams(), role_update_index=2)
    assert row["n_reputation"] == 1 and row["n_status"] == 1 and row["n_personal_utility"] == 2
    assert row["t"] == 7 and row["role_update_index"] == 2
    json.dumps(row)
