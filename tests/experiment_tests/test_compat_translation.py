"""
EXACT tests for compat.py: the config translation is a pure function of its
inputs, so nothing here is distributional. This is the layer where a porting
mistake is cheapest to catch.
"""
import numpy as np
import pytest

from experiments import compat
from model.config import (
    ActorRateDriverMode, Eq9Mode, LeaderUpdateMode, RewardModelKind, TrackingMode,
)

from .conftest import ACTIVE


def test_flat_kwargs_land_in_the_right_groups():
    cfg = compat.SystemConfig(
        num_agents=7, num_states=4, num_actions=3, num_time_steps=123,
        gamma=3.5, kappa=1.25, c_threshold=0.3, B_R=0.9, B_F=0.4, delta=0.05,
        M=2.0, u_0=0.2,
    )
    assert (cfg.dims.num_agents, cfg.dims.num_states, cfg.dims.num_actions) == (7, 4, 3)
    assert cfg.runtime.num_time_steps == 123
    assert cfg.algorithm.gamma == 3.5
    assert cfg.algorithm.kappa == 1.25
    assert cfg.algorithm.c_threshold == 0.3
    assert (cfg.algorithm.B_R, cfg.algorithm.B_F) == (0.9, 0.4)
    assert cfg.algorithm.delta == 0.05
    assert (cfg.algorithm.M, cfg.algorithm.u_0) == (2.0, 0.2)


def test_stepsize_bases_and_decays_match_the_original():
    """The old engine hardcoded the decays in step(); they must survive the move
    into config, or every learning rate silently changes."""
    cfg = compat.SystemConfig(alpha_pu_base=0.11, beta_status_base=0.22,
                              eta_v_base=0.33, eta_s_base=0.44, eta_J_base=0.55)
    s = cfg.stepsizes
    assert (s.alpha_pu.base, s.alpha_pu.decay) == (0.11, 0.01)
    assert (s.beta_status.base, s.beta_status.decay) == (0.22, 0.01)
    assert (s.eta_v.base, s.eta_v.decay) == (0.33, 0.01)
    assert (s.eta_s.base, s.eta_s.decay) == (0.44, 0.01)
    assert (s.eta_J.base, s.eta_J.decay) == (0.55, 0.01)
    assert (s.alpha_rate.base, s.alpha_rate.decay) == (0.01, 0.005)


@pytest.mark.parametrize("value", [k.value for k in RewardModelKind])
def test_reward_model_strings_map_to_enums(value):
    assert compat.SystemConfig(reward_model=value).reward.kind.value == value


@pytest.mark.parametrize("value", [m.value for m in Eq9Mode])
def test_eq9_mode_strings_map(value):
    assert compat.SystemConfig(eq9_averaging_mode=value).algorithm.eq9_averaging_mode.value == value


@pytest.mark.parametrize("value", [m.value for m in LeaderUpdateMode])
def test_leader_update_mode_strings_map(value):
    assert compat.SystemConfig(leader_update_mode=value).algorithm.leader_update_mode.value == value


@pytest.mark.parametrize("value", [m.value for m in ActorRateDriverMode])
def test_actor_rate_driver_mode_strings_map(value):
    assert compat.SystemConfig(actor_rate_driver_mode=value).algorithm.actor_rate_driver_mode.value == value


@pytest.mark.parametrize("value", ["full", "light"])
def test_tracking_mode_strings_map(value):
    assert compat.SystemConfig(tracking_mode=value).runtime.tracking_mode.value == value


def test_reward_params_all_route_through():
    cfg = compat.SystemConfig(
        reward_base_mu=0.6, reward_base_sigma=0.07, reward_agent_sigma=0.02,
        reward_clip_min=0.05, reward_clip_max=3.0, reward_good_value=1.5,
        reward_bad_value=0.2, reward_order_gap=0.03)
    r = cfg.reward
    assert (r.base_mu, r.base_sigma, r.agent_sigma) == (0.6, 0.07, 0.02)
    assert (r.clip_min, r.clip_max) == (0.05, 3.0)
    assert (r.good_value, r.bad_value, r.order_gap) == (1.5, 0.2, 0.03)


def test_schedule_params_route_through():
    cfg = compat.SystemConfig(role_update_s0=5, role_update_T_sequence=[10, 20],
                              role_update_base_interval=33,
                              fixed_role_update_interval=True,
                              role_update_epochs=[100, 200])
    s = cfg.schedule
    assert s.role_update_s0 == 5
    assert list(s.role_update_T_sequence) == [10, 20]
    assert s.role_update_base_interval == 33
    assert s.fixed_role_update_interval is True
    assert list(s.role_update_epochs) == [100, 200]


def test_unknown_kwarg_raises():
    """A typo in a ported harness must fail loudly. The old dataclass raised too."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        compat.SystemConfig(num_agents=5, gamm=2.0)


def test_every_kwarg_the_harnesses_use_is_accepted():
    """The union of SystemConfig kwargs across the four v1 harnesses."""
    compat.SystemConfig(**{
        "num_agents": 6, "num_states": 3, "num_actions": 2, "num_time_steps": 100,
        "M": 1.0, "u_0": 0.1, "actor_rate_driver_mode": "standard",
        "actor_rate_status_override_min_followers": 10,
        "gamma": 2.0, "kappa": 2.0, "c_threshold": 0.1, "B_R": 0.8, "B_F": 0.6,
        "delta": 0.1, "eq9_averaging_mode": "participants_only",
        "leader_update_mode": "participants_only_post_eq9",
        "alpha_pu_base": 0.05, "beta_status_base": 0.05, "eta_v_base": 0.1,
        "eta_s_base": 0.1, "eta_J_base": 0.05,
        "role_update_s0": 0, "role_update_T_sequence": [],
        "role_update_base_interval": 50, "fixed_role_update_interval": False,
        "role_update_epochs": [], "gossip_rate": 0.5, "gossip_alpha": 0.5,
        "tracking_mode": "full", "use_numpy_fast_path": False,
        "force_all_active_debug": False,
        "initial_actor_interaction_rate": 0.7,
        "initial_participant_interaction_rate": 0.7,
        "reward_model": "simple_preferred_action", "reward_base_mu": 0.5,
        "reward_base_sigma": 0.08, "reward_agent_sigma": 0.03,
        "reward_clip_min": 0.01, "reward_clip_max": 2.5,
        "reward_good_value": 1.0, "reward_bad_value": 0.1, "reward_order_gap": 0.02,
    })


# ---------------------------------------------------------------- results proxy

def test_results_supports_dict_and_attribute_access():
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    system.step()
    assert system.results["follower_counts"] == system.results.follower_counts
    assert "follower_counts" in system.results


def test_results_setdefault_append_works():
    """status_scaling does results.setdefault('role_update_times', []).append(t)."""
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    system.step()
    before = len(system.results["role_update_times"])
    system.results.setdefault("role_update_times", []).append(999)
    assert system.results["role_update_times"][-1] == 999
    assert len(system.results["role_update_times"]) == before + 1


def test_results_get_returns_default_for_missing_key():
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    assert system.results.get("not_a_field", "fallback") == "fallback"


def test_dict_of_results_works():
    """pu_scaling does `results = dict(system.results)`."""
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    system.step()
    d = dict(system.results)
    assert "follower_counts" in d and len(d["follower_counts"]) == 1


# ------------------------------------------------------------- legacy methods

def test_legacy_role_update_pair_runs():
    """The async harnesses call _update_roles_sequential then
    refresh_last_tracked_state as two separate steps."""
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    for _ in range(5):
        system.step()
    before = list(system.results["follower_counts"])
    system._update_roles_sequential([0, 1, 2])
    system.refresh_last_tracked_state()
    assert len(system.results["follower_counts"]) == len(before)


def test_refresh_leaves_activation_counts_alone():
    """refresh_last_tracked_state rewrites a SUBSET of fields — the activation
    counts are deliberately not among them."""
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    for _ in range(5):
        system.step()
    before = list(system.results["actor_counts"])
    system._update_roles_sequential([0, 1])
    system.refresh_last_tracked_state()
    assert system.results["actor_counts"] == before


def test_role_update_epoch_is_writable():
    """The async harnesses increment system.role_update_epoch themselves."""
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    assert system.role_update_epoch == 0
    system.role_update_epoch += 1
    assert system.role_update_epoch == 1


@pytest.mark.parametrize("enabler,flag", [
    ("enable_async_decision_audit", "async_audit"),
    ("enable_role_update_diagnostics", "role_update_diagnostics"),
    ("enable_small_n_trace_export", "dense_history"),
])
def test_audit_enablers_swap_in_a_full_recorder(enabler, flag):
    """These were post-construction methods in the old API and are constructor
    flags in the new one."""
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    getattr(system, enabler)()
    assert getattr(system.rec, flag) is True


def test_audit_enablers_compose():
    """perturbation_recovery calls more than one."""
    system = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
    system.enable_async_decision_audit()
    system.enable_role_update_diagnostics()
    assert system.rec.async_audit and system.rec.role_update_diagnostics


def test_same_global_seed_reproduces_the_run():
    """compat derives runtime.seed from the global stream, so a harness calling
    np.random.seed(s) still gets a deterministic run."""
    def run(seed):
        np.random.seed(seed)
        s = compat.MultiAgentSystem(compat.SystemConfig(**ACTIVE))
        for _ in range(20):
            s.step()
        return s.results["paper_welfare_all_agents"][-1]
    assert run(3) == run(3)
    assert run(3) != run(4)
