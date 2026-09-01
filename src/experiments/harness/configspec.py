"""
SystemConfig construction from CLI args plus a grid cell.

`make_config` was ~55 near-identical lines in each legacy harness. The only real
differences were which sweep variable got injected, which fields were read from
args versus hardcoded, and -- in one case -- a bug: reputation_status_scaling's
original `make_config` declared --c-threshold/--B-R/--B-F on the CLI and then
hardcoded all three, so those flags did nothing. Building the config from one
function with an explicit override dict makes that class of mistake visible.

Note the stepsize bases are identical across all four experiments
(alpha_pu=0.05, beta_status=0.05, eta_v=0.1, eta_s=0.1, eta_J=0.05) with decay
0.01, so they live here rather than in each experiment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from model.config import (
    ActorRateDriverMode,
    AlgorithmParams,
    Dimensions,
    Eq9Mode,
    LeaderUpdateMode,
    RewardModelKind,
    RewardParams,
    RuntimeParams,
    ScheduleParams,
    Stepsize,
    StepsizeParams,
    SystemConfig,
    TrackingMode,
)

from src.experiments.harness.schedule import async_role_interval_override

#: Every legacy harness used these bases with a 0.01 decay.
DEFAULT_STEPSIZES = StepsizeParams(
    alpha_pu=Stepsize(0.05, 0.01),
    beta_status=Stepsize(0.05, 0.01),
    eta_v=Stepsize(0.10, 0.01),
    eta_s=Stepsize(0.10, 0.01),
    eta_J=Stepsize(0.05, 0.01),
)


def make_config(
    args,
    *,
    mode: str,
    seed: int,
    overrides: Optional[Dict[str, Any]] = None,
) -> SystemConfig:
    """Build a SystemConfig.

    `overrides` is a flat dict of field names taking precedence over the
    corresponding CLI value. It is where a grid cell is injected (`gamma`,
    `kappa`, `num_states`, `reward_model`) and where an experiment states a
    deliberate deviation. Unknown keys raise, so a typo fails at startup.
    """
    o = dict(overrides or {})

    def pick(name: str, default: Any = None) -> Any:
        if name in o:
            return o.pop(name)
        return getattr(args, name, default)

    num_steps = int(pick("num_steps", None) or getattr(args, "num_steps_max", 0))

    dims = Dimensions(
        num_agents=int(pick("num_agents")),
        num_states=int(pick("num_states")),
        num_actions=int(pick("num_actions")),
    )

    algorithm = AlgorithmParams(
        gamma=float(pick("gamma", 0.0)),
        kappa=float(pick("kappa", 0.0)),
        c_threshold=float(pick("c_threshold")),
        B_R=float(pick("B_R")),
        B_F=float(pick("B_F")),
        delta=float(pick("delta")),
        M=float(pick("M", 1.0)),
        u_0=float(pick("u_0", 0.1)),
        gossip_rate=float(pick("gossip_rate", 0.5)),
        gossip_alpha=float(pick("gossip_alpha", 0.5)),
        initial_actor_interaction_rate=float(pick("initial_actor_rate")),
        initial_participant_interaction_rate=float(pick("initial_participant_rate")),
        actor_rate_status_override_min_followers=int(
            pick("actor_rate_status_override_min_followers")
        ),
        actor_rate_driver_mode=ActorRateDriverMode(pick("actor_rate_driver_mode")),
        eq9_averaging_mode=Eq9Mode(pick("eq9_averaging_mode")),
        leader_update_mode=LeaderUpdateMode(pick("leader_update_mode")),
    )

    reward_kwargs = dict(
        kind=RewardModelKind(pick("reward_model")),
        base_mu=float(pick("reward_base_mu")),
        base_sigma=float(pick("reward_base_sigma")),
        agent_sigma=float(pick("reward_agent_sigma")),
        clip_min=float(pick("reward_clip_min")),
        clip_max=float(pick("reward_clip_max")),
    )
    # Experiment D is the only one exposing these three.
    for extra, field in (
        ("reward_good_value", "good_value"),
        ("reward_bad_value", "bad_value"),
        ("reward_order_gap", "order_gap"),
    ):
        if extra in o or hasattr(args, extra):
            reward_kwargs[field] = float(pick(extra))
    reward = RewardParams(**reward_kwargs)

    base_interval, s0, t_seq, epochs = async_role_interval_override(args, mode)
    schedule = ScheduleParams(
        role_update_s0=int(o.pop("role_update_s0", s0)),
        role_update_T_sequence=list(o.pop("role_update_T_sequence", t_seq)),
        role_update_base_interval=int(o.pop("role_update_base_interval", base_interval)),
        fixed_role_update_interval=bool(pick("fixed_role_update_interval")),
        role_update_epochs=list(o.pop("role_update_epochs", epochs)),
    )

    runtime = RuntimeParams(
        seed=int(seed),
        tracking_mode=TrackingMode(pick("tracking_mode")),
        use_numpy_fast_path=bool(pick("numpy_fast_path")),
        force_all_active_debug=bool(pick("force_all_active_debug")),
        num_time_steps=num_steps,
    )

    if o:
        raise TypeError(f"make_config: unknown override keys {sorted(o)}")

    return SystemConfig(
        dims=dims,
        algorithm=algorithm,
        reward=reward,
        stepsizes=DEFAULT_STEPSIZES,
        runtime=runtime,
        schedule=schedule,
    )
