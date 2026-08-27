"""system.py — orchestration, plus the end-to-end invariants."""
import numpy as np
import pytest

from src.agent import AgentRole
from src.config import (
    AlgorithmParams, Dimensions, RewardModelKind, RewardParams, RuntimeParams,
    ScheduleParams, SystemConfig, TrackingMode,
)
from src.instrumentation import FullRecorder, NullRecorder
from src.system import MultiAgentSystem


def cfg(**runtime):
    r = dict(seed=7, num_time_steps=30, tracking_mode=TrackingMode.FULL)
    r.update(runtime)
    return SystemConfig(
        dims=Dimensions(num_agents=5, num_states=3, num_actions=2),
        runtime=RuntimeParams(**r),
        schedule=ScheduleParams(role_update_base_interval=10),
    )


def test_constructs():
    s = MultiAgentSystem(cfg())
    assert len(s.agents) == 5
    assert s.rep.s.shape == (5, 5)
    assert s.time_step == 0


def test_single_step_runs():
    MultiAgentSystem(cfg()).step()


def test_step_increments_time_and_tracks_once():
    s = MultiAgentSystem(cfg())
    s.step(); s.step()
    assert s.time_step == 2
    assert s.results.step_count() == 2


def test_simulate_returns_results_and_finalizes():
    s = MultiAgentSystem(cfg(num_time_steps=25))
    res = s.simulate()
    assert res.step_count() == 25
    assert res.final_roles is not None and res.final_followers is not None


def test_simulate_prints_nothing(capsys):
    MultiAgentSystem(cfg(num_time_steps=5)).simulate()
    assert capsys.readouterr().out == ""


def test_same_seed_reproduces_the_whole_trajectory():
    a = MultiAgentSystem(cfg()).simulate()
    b = MultiAgentSystem(cfg()).simulate()
    assert a.paper_welfare_all_agents == b.paper_welfare_all_agents
    assert a.follower_counts == b.follower_counts
    assert a.role_label_history == b.role_label_history


def test_different_seeds_diverge():
    a = MultiAgentSystem(cfg(seed=1)).simulate()
    b = MultiAgentSystem(cfg(seed=2)).simulate()
    assert a.paper_welfare_all_agents != b.paper_welfare_all_agents


def test_global_rng_state_does_not_affect_the_run():
    """The tripwire for a surviving np.random.* call anywhere in the package."""
    np.random.seed(1)
    a = MultiAgentSystem(cfg()).simulate().follower_counts
    np.random.seed(999)
    b = MultiAgentSystem(cfg()).simulate().follower_counts
    assert a == b


def test_light_mode_skips_full_only_fields():
    res = MultiAgentSystem(cfg(tracking_mode=TrackingMode.LIGHT, num_time_steps=10)).simulate()
    assert res.step_count() == 10
    assert res.norm_consensus == []
    assert len(res.role_label_history) == 10, "role labels are written in BOTH modes"


def test_results_validate_after_a_run():
    MultiAgentSystem(cfg(num_time_steps=20)).simulate().validate()


def test_follow_graph_is_consistent_after_a_run():
    s = MultiAgentSystem(cfg(num_time_steps=40))
    s.simulate()
    for i, a in enumerate(s.agents):
        if a.state.following is not None:
            assert a.state.following != i
            assert i in s.agents[a.state.following].state.followers
        for f in a.state.followers:
            assert s.agents[f].state.following == i


def test_actor_rates_stay_in_budget():
    s = MultiAgentSystem(cfg(num_time_steps=40))
    s.simulate()
    M = s.config.algorithm.M
    assert all(0.0 <= a.state.actor_interaction_rate <= M for a in s.agents)


def test_reputation_state_stays_finite():
    s = MultiAgentSystem(cfg(num_time_steps=40))
    s.simulate()
    assert np.isfinite(s.rep.s).all() and np.isfinite(s.rep.v).all()


def test_payoff_history_has_one_entry_per_activation():
    """The duplicate append at line 264 doubled entries for PU agents."""
    s = MultiAgentSystem(cfg(num_time_steps=30, force_all_active_debug=True))
    s.simulate()
    for a in s.agents:
        assert len(a.state.payoff_history) == 30


def test_role_updates_fire_on_schedule():
    res = MultiAgentSystem(cfg(num_time_steps=45)).simulate()
    assert res.role_update_times == [10, 21, 33]


def test_force_all_active_activates_everyone():
    s = MultiAgentSystem(cfg(force_all_active_debug=True))
    s.step()
    assert s.results.actor_counts[-1] == 5
    assert s.results.participant_counts[-1] == 5


@pytest.mark.parametrize("kind", list(RewardModelKind))
def test_runs_under_every_reward_model(kind):
    c = cfg(num_time_steps=10)
    c.reward.model = kind
    MultiAgentSystem(c).simulate()


def test_full_recorder_collects_audit_rows():
    s = MultiAgentSystem(cfg(num_time_steps=25), rec=FullRecorder(role_update_diagnostics=True))
    res = s.simulate()
    assert len(s.rec.rows) > 0
    assert len(res.role_update_diagnostics) == len(res.role_update_times)


def test_dense_history_populates_the_dense_fields():
    s = MultiAgentSystem(cfg(num_time_steps=5), rec=FullRecorder(dense_history=True))
    res = s.simulate()
    assert len(res.dense_reputation_history) == 5
    assert res.dense_reputation_history[0].shape == (5, 5)


def test_null_recorder_leaves_audit_fields_empty():
    res = MultiAgentSystem(cfg(num_time_steps=20), rec=NullRecorder()).simulate()
    assert res.role_update_diagnostics == []
    assert res.dense_reputation_history == []


def test_update_roles_wrapper_exists_for_async_harnesses():
    """The three sweep harnesses called _update_roles_sequential(candidates) on
    the system; they need a public entry point after the split."""
    s = MultiAgentSystem(cfg())
    assert hasattr(s, "update_roles")


def test_common_random_numbers_across_a_parameter_change():
    """Activation draws must be indexed by (seed, agent, t), not by how many
    agents happened to activate, or paired sweep comparisons break."""
    c1, c2 = cfg(num_time_steps=1), cfg(num_time_steps=1)
    c2.algorithm.gamma = 5.0
    a, b = MultiAgentSystem(c1), MultiAgentSystem(c2)
    a.step(); b.step()
    assert a.results.actor_counts == b.results.actor_counts
