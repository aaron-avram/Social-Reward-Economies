"""
Legacy adapter: presents the pre-refactor engine API on top of `model/`.

Every old experiment harness imported exactly one line:

    from src.code_debugged import AgentRole, MultiAgentSystem, SystemConfig

The ported copies change that line to:

    from experiments.compat import AgentRole, MultiAgentSystem, SystemConfig

and nothing else. That keeps the port diff-free so the experiment-level output
comparison tests measure the ENGINE change, not a hand-translation of 8,000 lines.

THIS FILE IS TEMPORARY. Once the output tests pass, the unified harness should use
the real API directly and this module should be deleted. Everything it does is a
translation, and each translation is a place a future reader could be misled about
what the engine actually offers.
"""

from dataclasses import replace
from typing import Any, Iterable, Optional

import numpy as np

from model.agent import AgentRole  # re-exported: the harnesses import it from here
from model.config import (
    ActorRateDriverMode, AlgorithmParams, Dimensions, Eq9Mode, LeaderUpdateMode,
    RewardModelKind, RewardParams, RuntimeParams, ScheduleParams, Stepsize,
    StepsizeParams, SystemConfig as _SystemConfig, TrackingMode,
)
from model.instrumentation import FullRecorder, NullRecorder
from model.system import MultiAgentSystem as _MultiAgentSystem

__all__ = ["AgentRole", "MultiAgentSystem", "SystemConfig"]

# The old stepsize decays were hardcoded in step(); the new config carries them.
_DECAY = {
    "alpha_pu": 0.01, "beta_status": 0.01, "eta_v": 0.01,
    "eta_s": 0.01, "eta_J": 0.01,
}


def SystemConfig(**kw: Any) -> _SystemConfig:  # noqa: N802 - mimics the old class
    """
    Build a nested SystemConfig from the old flat keyword arguments.

    Accepts every kwarg the four harnesses pass. Unknown keys raise, so a typo in a
    ported harness fails loudly instead of being silently ignored (the old
    dataclass would have raised too).
    """
    kw = dict(kw)

    def take(name, default=None):
        return kw.pop(name, default)

    dims = Dimensions(
        num_agents=int(take("num_agents", 6)),
        num_states=int(take("num_states", 3)),
        num_actions=int(take("num_actions", 2)),
    )

    algorithm = AlgorithmParams(
        gamma=float(take("gamma", 2.0)),
        kappa=float(take("kappa", 2.0)),
        c_threshold=float(take("c_threshold", 0.1)),
        B_R=float(take("B_R", 0.8)),
        B_F=float(take("B_F", 0.6)),
        delta=float(take("delta", 0.1)),
        M=float(take("M", 1.0)),
        u_0=float(take("u_0", 0.1)),
        gossip_rate=float(take("gossip_rate", 0.5)),
        gossip_alpha=float(take("gossip_alpha", 0.5)),
        initial_actor_interaction_rate=float(take("initial_actor_interaction_rate", 0.7)),
        initial_participant_interaction_rate=float(
            take("initial_participant_interaction_rate", 0.7)),
        actor_rate_status_override_min_followers=int(
            take("actor_rate_status_override_min_followers", 10)),
        actor_rate_driver_mode=ActorRateDriverMode(take("actor_rate_driver_mode", "standard")),
        eq9_averaging_mode=Eq9Mode(take("eq9_averaging_mode", "participants_only")),
        leader_update_mode=LeaderUpdateMode(
            take("leader_update_mode", "participants_only_post_eq9")),
    )

    reward = RewardParams(
        kind=RewardModelKind(take("reward_model", "simple_preferred_action")),
        base_mu=float(take("reward_base_mu", 0.5)),
        base_sigma=float(take("reward_base_sigma", 0.08)),
        agent_sigma=float(take("reward_agent_sigma", 0.03)),
        clip_min=float(take("reward_clip_min", 0.01)),
        clip_max=float(take("reward_clip_max", 2.5)),
        good_value=float(take("reward_good_value", 1.0)),
        bad_value=float(take("reward_bad_value", 0.1)),
        order_gap=float(take("reward_order_gap", 0.02)),
        consensus_high=float(take("reward_consensus_high", 0.95)),
        consensus_low=float(take("reward_consensus_low", 0.45)),
        welfare_high=float(take("reward_welfare_high", 1.05)),
        welfare_low=float(take("reward_welfare_low", 0.35)),
        lambda_min=float(take("reward_lambda_min", 0.55)),
        lambda_max=float(take("reward_lambda_max", 0.85)),
    )

    stepsizes = StepsizeParams(
        alpha_pu=Stepsize(float(take("alpha_pu_base", 0.05)), _DECAY["alpha_pu"]),
        beta_status=Stepsize(float(take("beta_status_base", 0.10)), _DECAY["beta_status"]),
        eta_v=Stepsize(float(take("eta_v_base", 0.10)), _DECAY["eta_v"]),
        eta_s=Stepsize(float(take("eta_s_base", 0.10)), _DECAY["eta_s"]),
        eta_J=Stepsize(float(take("eta_J_base", 0.05)), _DECAY["eta_J"]),
        alpha_rate=Stepsize(0.01, 0.005),
    )

    runtime = RuntimeParams(
        seed=int(take("seed", 0)),
        tracking_mode=TrackingMode(take("tracking_mode", "full")),
        use_numpy_fast_path=bool(take("use_numpy_fast_path", False)),
        force_all_active_debug=bool(take("force_all_active_debug", False)),
        num_time_steps=int(take("num_time_steps", 2000)),
    )

    schedule = ScheduleParams(
        role_update_s0=int(take("role_update_s0", 0)),
        role_update_T_sequence=list(take("role_update_T_sequence", []) or []),
        role_update_base_interval=int(take("role_update_base_interval", 50)),
        fixed_role_update_interval=bool(take("fixed_role_update_interval", False)),
        role_update_epochs=list(take("role_update_epochs", []) or []),
    )

    if kw:
        raise TypeError(f"SystemConfig() got unexpected keyword arguments: {sorted(kw)}")

    return _SystemConfig(dims=dims, algorithm=algorithm, reward=reward,
                         stepsizes=stepsizes, runtime=runtime, schedule=schedule)


class _ResultsProxy:
    """
    Dict-like view over SimulationResults.

    The harnesses do results['key'], .get(), .setdefault(...).append(...), and
    dict(system.results). Attribute access still works, so new code can use either.
    """

    def __init__(self, results):
        self._r = results
        self._extra: dict[str, Any] = {}

    def __getitem__(self, key):
        if hasattr(self._r, key):
            return getattr(self._r, key)
        if key in self._extra:
            return self._extra[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if hasattr(self._r, key):
            setattr(self._r, key, value)
        else:
            self._extra[key] = value

    def __contains__(self, key):
        return hasattr(self._r, key) or key in self._extra

    def __getattr__(self, name):
        return getattr(self._r, name)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def keys(self):
        from dataclasses import fields
        return [f.name for f in fields(self._r)] + list(self._extra)

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __len__(self):
        return len(self.keys())


class MultiAgentSystem(_MultiAgentSystem):  # noqa: N801 - mimics the old class
    """
    The new engine wearing the old method names.

    Differences the harnesses cannot see, but which matter for interpreting results:

      * RNG. The old engine drew everything from the global stream, so
        np.random.seed(s) at the top of run_single() determined the trajectory.
        The new engine uses config.runtime.seed and five spawned substreams; the
        global seed no longer has any effect. This adapter copies the global
        stream's current position into runtime.seed at construction, so a harness
        that calls np.random.seed(s) still gets a deterministic, s-dependent run —
        but NOT the same run as before. Output comparisons must be distributional.

      * The softmax epsilon. The old get_softmax_policy divided by (sum + 1e-8);
        the new one normalises properly. Welfare differs by ~1e-8.
    """

    def __init__(self, config, rec=None):
        # Derive the engine seed from the global RNG, so np.random.seed(s) in a
        # harness still produces an s-dependent (if different) trajectory.
        if config.runtime.seed == 0:
            derived = int(np.random.randint(0, 2**31 - 1))
            config = replace(config, runtime=replace(config.runtime, seed=derived))
        super().__init__(config, rec if rec is not None else NullRecorder())
        self.results = _ResultsProxy(self.results)
        self.role_update_epoch = 0

    # --- audit enablers: methods in the old API, recorder flags in the new one ---

    def _ensure_full_recorder(self, **flags) -> None:
        current = self.rec if isinstance(self.rec, FullRecorder) else FullRecorder(
            async_audit=False, role_update_diagnostics=False, dense_history=False)
        for name, value in flags.items():
            setattr(current, name, value)
        self.rec = current

    def enable_async_decision_audit(self) -> None:
        self._ensure_full_recorder(async_audit=True)

    def enable_role_update_diagnostics(self) -> None:
        self._ensure_full_recorder(role_update_diagnostics=True)

    def enable_small_n_trace_export(self) -> None:
        self._ensure_full_recorder(dense_history=True)

    def set_decision_audit_preleader(self, agent_id: Optional[int]) -> None:
        self._ensure_full_recorder()
        self.rec.preleader_id = None if agent_id is None or int(agent_id) < 0 else int(agent_id)

    def get_async_decision_audit_rows(self) -> list:
        return list(getattr(self.rec, "rows", []))

    def get_role_update_diagnostic_rows(self) -> list:
        return list(self.results.role_update_diagnostics)

    def get_true_reputation_checkpoint_rows(self) -> list:
        return list(self.results.true_reputation_checkpoints)

    def get_estimate_consensus_checkpoint_rows(self) -> list:
        return list(self.results.estimate_consensus_checkpoints)

    def get_rate_audit_checkpoint_rows(self) -> list:
        return list(self.results.rate_audit_checkpoints)

    # --- async role updates ---

    def _update_roles_sequential(self, update_candidates: Optional[Iterable[int]] = None) -> None:
        """Old name. The refresh is a separate call in the old API, so it is
        suppressed here and performed by refresh_last_tracked_state() below —
        preserving the original call order exactly."""
        self.update_roles(update_candidates, refresh=False)

    def refresh_last_tracked_state(self) -> None:
        self.results.overwrite_last(self._build_refresh_record())

    def simulate(self, *args, **kwargs):
        """The old simulate() printed a summary and returned a dict. Harnesses
        wrap it in redirect_stdout and read results by key, so returning the
        proxy keeps both behaviours working."""
        super().simulate(*args, **kwargs)
        return self.results
