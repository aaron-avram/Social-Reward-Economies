"""roleupdate.py — Section 7's three steps, plus the epoch schedule."""
import numpy as np
import pytest

from model.agent import Agent, AgentRole
from model.config import AlgorithmParams, Dimensions, ScheduleParams
from model.instrumentation import NullRecorder
from model import roleupdate as RU
from model.reputation import NO_LEADER


DIMS = Dimensions(num_agents=5, num_states=3, num_actions=2)


def make_agents(n=5):
    return [Agent(i, AlgorithmParams(), Dimensions(num_agents=n, num_states=3, num_actions=2),
                  np.random.default_rng(0)) for i in range(n)]


def signals(n, target_rep=0.0, pu=0.0, status=0.0, target=None, roles=None):
    return {
        i: RU.AgentSignals(
            role=(roles[i] if roles else AgentRole.PERSONAL_UTILITY),
            target=(target[i] if target is not None else (i + 1) % n),
            target_rep=target_rep, estimated_reward_pu=pu,
            estimated_reward_status=status,
            rep_row=np.zeros(n),
        )
        for i in range(n)
    }


def blank_state(n=5):
    return RU.RoleUpdateState(P=set(range(n)), R=set(), S=set(),
                              followers={i: set() for i in range(n)},
                              following={i: None for i in range(n)})


# ------------------------------------------------------- working state

def test_from_agents_partitions_by_role():
    ags = make_agents()
    ags[1].state.role = AgentRole.REPUTATION
    ags[2].state.role = AgentRole.STATUS
    st = RU.RoleUpdateState.from_agents(ags)
    assert st.P == {0, 3, 4} and st.R == {1} and st.S == {2}


def test_from_agents_snapshots_following_and_followers():
    ags = make_agents()
    ags[0].state.following = 1
    ags[1].state.followers = {0}
    st = RU.RoleUpdateState.from_agents(ags)
    assert st.following[0] == 1 and st.followers[1] == {0}
    st.followers[1].add(9)
    assert ags[1].state.followers == {0}, "must be a copy, not an alias"


def test_apply_writes_role_following_and_followers():
    ags = make_agents()
    st = blank_state()
    st.P.discard(2); st.R.add(2); st.following[2] = 0; st.followers[0] = {2}
    st.apply(ags)
    assert ags[2].state.role is AgentRole.REPUTATION
    assert ags[2].state.following == 0
    assert ags[0].state.followers == {2}


def test_apply_maps_the_partition_to_the_right_roles():
    """P/R/S -> PU/REPUTATION/STATUS. Swapping two branches is invisible until
    the role counts come out wrong."""
    ags = make_agents(3)
    st = RU.RoleUpdateState(P={0}, R={1}, S={2},
                            followers={i: set() for i in range(3)},
                            following={i: None for i in range(3)})
    st.apply(ags)
    assert [a.state.role for a in ags] == [
        AgentRole.PERSONAL_UTILITY, AgentRole.REPUTATION, AgentRole.STATUS]


def test_detach_removes_from_every_follower_set():
    st = blank_state()
    st.followers[0] = {3}; st.followers[1] = {3}
    st.detach(3)
    assert st.followers[0] == set() and st.followers[1] == set()


def test_validate_catches_a_broken_partition():
    st = blank_state()
    st.R.add(0)          # 0 now in both P and R
    with pytest.raises(AssertionError):
        st.validate(5)


def test_resolve_updatable_none_means_all():
    assert RU.resolve_updatable(None, 4) == {0, 1, 2, 3}


def test_resolve_updatable_drops_out_of_range():
    assert RU.resolve_updatable([-1, 0, 2, 99], 4) == {0, 2}


# ------------------------------------------------------------- step 1

def test_step1_follows_when_condition_met():
    st = blank_state()
    sig = signals(5, target_rep=1.0, pu=0.0, target=[1] * 5)
    RU.step1_reputation(st, sig, {0}, 5, AlgorithmParams(gamma=2.0),
                        np.random.default_rng(0), NullRecorder())
    assert st.following[0] == 1
    assert 0 in st.followers[1]
    assert 0 in st.R and 0 not in st.P


def test_step1_declines_below_threshold():
    st = blank_state()
    sig = signals(5, target_rep=0.1, pu=0.0, target=[1] * 5)
    RU.step1_reputation(st, sig, {0}, 5, AlgorithmParams(gamma=2.0),
                        np.random.default_rng(0), NullRecorder())
    assert st.following[0] is None and 0 not in st.R


def test_step1_condition_compares_against_max_of_threshold_and_pu():
    """gamma*rep = 1.8 clears B_R = 0.8 but not J^pu = 2.0."""
    st = blank_state()
    sig = signals(5, target_rep=0.9, pu=2.0, target=[1] * 5)
    RU.step1_reputation(st, sig, {0}, 5, AlgorithmParams(gamma=2.0),
                        np.random.default_rng(0), NullRecorder())
    assert st.following[0] is None


def test_step1_hysteresis_uses_B_F_for_current_followers():
    """gamma*rep = 1.4 clears B_F=0.6 but not B_R=0.8 ... it clears both here;
    use a value between them: rep=0.35, gamma=2 -> 0.7."""
    p = AlgorithmParams(gamma=2.0, B_R=0.8, B_F=0.6)
    sig = signals(5, target_rep=0.35, pu=0.0, target=[1] * 5)

    fresh = blank_state()
    RU.step1_reputation(fresh, sig, {0}, 5, p, np.random.default_rng(0), NullRecorder())
    assert fresh.following[0] is None, "0.7 must not clear B_R=0.8"

    already = blank_state()
    already.P.discard(0); already.R.add(0); already.following[0] = 1
    already.followers[1] = {0}
    RU.step1_reputation(already, sig, {0}, 5, p, np.random.default_rng(0), NullRecorder())
    assert already.following[0] == 1, "0.7 must clear B_F=0.6 and keep following"


def test_step1_redirects_away_from_a_follower_target():
    """[ROLE-3]: if the target is itself following someone, follow its leader."""
    st = blank_state()
    st.following[1] = 2
    st.followers[2] = {1}
    sig = signals(5, target_rep=1.0, target=[1] * 5)
    RU.step1_reputation(st, sig, {0}, 5, AlgorithmParams(gamma=2.0),
                        np.random.default_rng(0), NullRecorder())
    assert st.following[0] == 2, "should follow the leader's leader, not the follower"


def test_step1_blocks_self_follow_after_redirect():
    st = blank_state()
    st.following[1] = 0
    st.followers[0] = {1}
    sig = signals(5, target_rep=1.0, target=[1] * 5)
    RU.step1_reputation(st, sig, {0}, 5, AlgorithmParams(gamma=2.0),
                        np.random.default_rng(0), NullRecorder())
    assert st.following[0] != 0


def test_step1_rehomes_existing_followers_to_the_new_leader():
    """[ROLE-5]: agent 0's followers must move to 0's new leader AND appear in
    that leader's follower set."""
    st = blank_state()
    st.followers[0] = {3, 4}
    st.following[3] = st.following[4] = 0
    sig = signals(5, target_rep=1.0, target=[1] * 5)
    RU.step1_reputation(st, sig, {0,1,2,3,4}, 5, AlgorithmParams(gamma=2.0),
                        np.random.default_rng(0), NullRecorder())
    assert st.following[3] == 1 and st.following[4] == 1
    assert {3, 4} <= st.followers[1]
    assert st.followers[0] == set()


def test_step1_only_considers_followerless_agents():
    st = blank_state()
    st.followers[0] = {2}
    sig = signals(5, target_rep=1.0, target=[1] * 5)
    RU.step1_reputation(st, sig, {0}, 5, AlgorithmParams(gamma=2.0),
                        np.random.default_rng(0), NullRecorder())
    assert st.following[0] is None


def test_step1_consumes_the_order_stream():
    a, b = np.random.default_rng(0), np.random.default_rng(0)
    RU.step1_reputation(blank_state(), signals(5), {0, 1, 2, 3, 4}, 5,
                        AlgorithmParams(), a, NullRecorder())
    assert a.bit_generator.state != b.bit_generator.state


# ------------------------------------------------------------- step 2

def test_step2_takes_status_above_threshold():
    st = blank_state()
    st.followers[0] = {1, 2, 3}
    sig = signals(5, pu=0.1, status=1.0)
    RU.step2_status(st, sig, {0}, AlgorithmParams(kappa=2.0, c_threshold=0.5), 5,
                    NullRecorder())
    assert 0 in st.S and 0 not in st.P and 0 not in st.R


def test_step2_requires_enough_followers():
    st = blank_state()
    st.followers[0] = {1}
    sig = signals(5, pu=0.1, status=1.0)
    RU.step2_status(st, sig, {0}, AlgorithmParams(kappa=2.0, c_threshold=0.8), 5,
                    NullRecorder())
    assert 0 not in st.S


def test_step2_requires_status_to_beat_pu():
    st = blank_state()
    st.followers[0] = {1, 2, 3, 4}
    sig = signals(5, pu=10.0, status=0.1)
    RU.step2_status(st, sig, {0}, AlgorithmParams(kappa=2.0, c_threshold=0.2), 5,
                    NullRecorder())
    assert 0 not in st.S


def test_step2_drops_its_own_leader():
    st = blank_state()
    st.followers[0] = {1, 2, 3}
    st.following[0] = 4
    st.followers[4] = {0}
    sig = signals(5, pu=0.1, status=1.0)
    RU.step2_status(st, sig, {0}, AlgorithmParams(kappa=2.0, c_threshold=0.5), 5,
                    NullRecorder())
    assert st.following[0] is None and 0 not in st.followers[4]


def test_status_threshold_is_ceil():
    assert RU.status_threshold(0.1, 15) == 2
    assert RU.status_threshold(0.5, 4) == 2
    assert RU.status_threshold(0.0, 10) == 0


# ------------------------------------------------------------- step 3

def test_step3_sends_unassigned_agents_to_pu():
    st = blank_state()
    st.P.clear()
    st.following[0] = 1
    RU.step3_fallback_pu(st, {0}, NullRecorder())
    assert 0 in st.P and st.following[0] is None


def test_step3_leaves_R_and_S_alone():
    st = blank_state()
    st.P.clear(); st.R.add(0); st.S.add(1)
    st.following[0] = 2
    RU.step3_fallback_pu(st, {0, 1}, NullRecorder())
    assert st.following[0] == 2 and 0 in st.R and 1 in st.S


# ------------------------------------------------------------- signals

def test_collect_signals_reads_the_reputation_matrix():
    ags = make_agents(4)
    s = np.arange(16, dtype=float).reshape(4, 4)
    L = np.array([2, 0, 1, 3])
    sig = RU.collect_signals(ags, s, L)
    assert sig[0].target == 2
    assert sig[0].target_rep == pytest.approx(s[0, 2])


def test_collect_signals_zero_rep_for_unresolved_leader():
    """L == -1 must give 0.0, NOT s[i, -1] (the last agent's column)."""
    ags = make_agents(4)
    s = np.zeros((4, 4))
    s[:, 3] = 99.0                       # column -1
    L = np.full(4, NO_LEADER)
    sig = RU.collect_signals(ags, s, L)
    assert sig[0].target_rep == 0.0


def test_collect_signals_exposes_the_full_reputation_row():
    """instrumentation.role_update_begin reads sig.rep_row for the currently
    followed agent and the pre-perturbation leader."""
    ags = make_agents(4)
    s = np.arange(16, dtype=float).reshape(4, 4)
    sig = RU.collect_signals(ags, s, np.zeros(4, dtype=int))
    assert np.allclose(sig[0].rep_row, s[0, :])


# ------------------------------------------------------- orchestration

def test_update_roles_sequential_runs_end_to_end():
    ags = make_agents(5)
    s = np.zeros((5, 5)); s[:, 1] = 1.0
    L = np.ones(5, dtype=int)
    RU.update_roles_sequential(ags, s, L, AlgorithmParams(gamma=2.0),
                               np.random.default_rng(0))
    st = RU.RoleUpdateState.from_agents(ags)
    st.validate(5)


def test_update_roles_sequential_empty_candidates_is_a_noop():
    ags = make_agents(5)
    before = [a.state.role for a in ags]
    RU.update_roles_sequential(ags, np.zeros((5, 5)), np.zeros(5, dtype=int),
                               AlgorithmParams(), np.random.default_rng(0),
                               update_candidates=[])
    assert [a.state.role for a in ags] == before


def test_stale_status_is_cleared_before_step1():
    """[STATUS-2]: a zero-follower STATUS agent must not persist."""
    ags = make_agents(5)
    ags[0].state.role = AgentRole.STATUS
    RU.update_roles_sequential(ags, np.zeros((5, 5)), np.full(5, NO_LEADER),
                               AlgorithmParams(), np.random.default_rng(0))
    assert ags[0].state.role is not AgentRole.STATUS


def test_follow_graph_stays_consistent_after_an_update():
    ags = make_agents(6)
    s = np.zeros((6, 6)); s[:, 2] = 1.0
    L = np.full(6, 2)
    RU.update_roles_sequential(ags, s, L, AlgorithmParams(gamma=2.0),
                               np.random.default_rng(1))
    for i, a in enumerate(ags):
        if a.state.following is not None:
            assert i in ags[a.state.following].state.followers
        for f in a.state.followers:
            assert ags[f].state.following == i


def test_no_agent_follows_itself():
    ags = make_agents(6)
    s = np.zeros((6, 6)); s[:, 0] = 1.0
    RU.update_roles_sequential(ags, s, np.zeros(6, dtype=int),
                               AlgorithmParams(gamma=2.0), np.random.default_rng(2))
    assert all(a.state.following != i for i, a in enumerate(ags))


# ---------------------------------------------------------- schedule

def test_explicit_epochs_fire_once_each_and_then_exhaust():
    s = RU.RoleUpdateSchedule(ScheduleParams(role_update_epochs=[10, 20]))
    assert [s.due_count(t) for t in (5, 10, 15, 20, 25, 100)] == [0, 1, 0, 1, 0, 0]


def test_t_sequence_accumulates_from_s0():
    s = RU.RoleUpdateSchedule(ScheduleParams(role_update_s0=5,
                                             role_update_T_sequence=[10, 20]))
    assert s.epochs == [15, 35]


def test_t_sequence_takes_precedence_over_epoch_list():
    s = RU.RoleUpdateSchedule(ScheduleParams(role_update_T_sequence=[10],
                                             role_update_epochs=[999]))
    assert s.epochs == [10]


def test_fixed_interval_fires_on_multiples():
    s = RU.RoleUpdateSchedule(ScheduleParams(role_update_base_interval=10,
                                             fixed_role_update_interval=True))
    assert [t for t in range(1, 45) if s.due_count(t)] == [10, 20, 30, 40]


def test_increasing_intervals_match_the_original_counter():
    """The original advances _next_role_update_time by max(base, base*(1+n*0.1))
    after each epoch: 50, 105, 165, 230, 300, 375."""
    s = RU.RoleUpdateSchedule(ScheduleParams(role_update_base_interval=50))
    assert [t for t in range(1, 400) if s.due_count(t)] == [50, 105, 165, 230, 300, 375]


def test_due_count_returns_an_int():
    s = RU.RoleUpdateSchedule(ScheduleParams(role_update_base_interval=10))
    assert isinstance(s.due_count(1), int)


def test_due_count_catches_up_on_a_time_jump():
    s = RU.RoleUpdateSchedule(ScheduleParams(role_update_epochs=[1, 2, 3]))
    assert s.due_count(10) == 3
