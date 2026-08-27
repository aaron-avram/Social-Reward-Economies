"""
Typed per-run results.

Replaces the 30-key string dict declared at 553-590 of code_debugged.py.

Four discrepancies in the original that this file fixes, all verified against source:

  1. SEVEN+ fields are created only by setdefault and never appear in the
     declaration: online_active_actor_payoff_sum (2399), social_welfare (2410),
     status_counts (2415), pu_counts (2418), rep_counts (2421),
     role_update_diagnostics (2425), and three *_checkpoints (2434-2441).
     Declared schema != runtime schema.

  2. THREE fields are set by simulate() as scalars, not lists: final_roles,
     final_followers, opinion_leader (2536-2538). A dict of "histories" that also
     holds three non-histories is a trap for anything iterating it.

  3. refresh_last_tracked_state (1314-1404) is NOT a full mirror of _track_results.
     It overwrites ~21 fields and leaves ~10 alone (actor_counts,
     participant_counts, online_active_actor_payoff_sum, norm_consensus,
     expected_utilities, actor_rates, roles_history, actual_payoffs,
     role_update_times). Marked per field below with NOT where it is skipped.
     Preserve this asymmetry — it is live behaviour.

  4. Field types are not what the names suggest:
       expected_utilities  -> dict[int, float], not float    (2494-2498)
       roles_history       -> list[AgentRole], not list[str] (2500)
       role_update_times   -> RAGGED; appended only on update epochs (2413)
"""

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import numpy as np

from model.agent import AgentRole
from model.config import SCHEMA_VERSION


# ============================================================================
# Per-step record
# ============================================================================

@dataclass
class StepRecord:
    """
    One timestep's worth of tracked state, built once and then either appended or
    used to overwrite the last entry.

    A None field means "not collected this step". Mode gating happens at BUILD
    time, in system._track, so the methods below never inspect TrackingMode.
    """
    t: int

    # --- core, both modes ---
    follower_counts: list[int]
    actor_count: int
    participant_count: int
    online_active_actor_payoff_sum: float
    paper_welfare_all_agents: float
    paper_welfare_followers_only: float
    status_count: int
    pu_count: int
    rep_count: int
    role_label: list[str]

    # --- compact histories (FULL, or LIGHT with compact debug enabled) ---
    estimated_reward_pu: Optional[list[float]] = None
    estimated_reward_rep: Optional[list[float]] = None
    estimated_reward_status: Optional[list[float]] = None
    actor_interaction_rate: Optional[list[float]] = None
    selected_reputation: Optional[list[float]] = None
    weighted_selected_reputation: Optional[list[float]] = None
    highest_rep_agent: Optional[list[int]] = None
    following: Optional[list[int]] = None

    # --- FULL only ---
    norm_consensus: Optional[float] = None
    expected_utilities: Optional[dict[int, float]] = None
    actor_rates: Optional[list[float]] = None
    roles: Optional[list[AgentRole]] = None
    actual_payoffs: Optional[dict[int, float]] = None

    # --- dense, opt-in (recorder.wants_dense_history) ---
    dense_reputation: Optional[np.ndarray] = None
    dense_personal_benefit: Optional[np.ndarray] = None
    true_reputation: Optional[np.ndarray] = None
    true_reputation_rank: Optional[np.ndarray] = None
    true_reputation_theta: Optional[np.ndarray] = None
    true_reputation_sum_expected: Optional[np.ndarray] = None
    active_actor_ids: Optional[list[int]] = None
    active_participant_ids: Optional[list[int]] = None
    observed_utility_matrix: Optional[np.ndarray] = None
    eta_v: Optional[float] = None
    gossip_target_ids: Optional[list[int]] = None
    averaging_agent_ids: Optional[list[int]] = None
    avg_s_by_target: Optional[dict[int, float]] = None
    delta_v_matrix: Optional[np.ndarray] = None


# StepRecord field -> SimulationResults field. Names differ because the original
# dict keys are what the three sweep harnesses read; keep them.
_FIELD_MAP = {
    "follower_counts": "follower_counts",
    "actor_count": "actor_counts",
    "participant_count": "participant_counts",
    "online_active_actor_payoff_sum": "online_active_actor_payoff_sum",
    "paper_welfare_all_agents": "paper_welfare_all_agents",
    "paper_welfare_followers_only": "paper_welfare_followers_only",
    "status_count": "status_counts",
    "pu_count": "pu_counts",
    "rep_count": "rep_counts",
    "role_label": "role_label_history",
    "estimated_reward_pu": "estimated_reward_pu_history",
    "estimated_reward_rep": "estimated_reward_rep_history",
    "estimated_reward_status": "estimated_reward_status_history",
    "actor_interaction_rate": "actor_interaction_rate_history",
    "selected_reputation": "selected_reputation_history",
    "weighted_selected_reputation": "weighted_selected_reputation_history",
    "highest_rep_agent": "highest_rep_agent_history",
    "following": "following_history",
    "norm_consensus": "norm_consensus",
    "expected_utilities": "expected_utilities",
    "actor_rates": "actor_rates",
    "roles": "roles_history",
    "actual_payoffs": "actual_payoffs",
    "dense_reputation": "dense_reputation_history",
    "dense_personal_benefit": "dense_personal_benefit_history",
    "true_reputation": "true_reputation_history",
    "true_reputation_rank": "true_reputation_rank_history",
    "true_reputation_theta": "true_reputation_theta_history",
    "true_reputation_sum_expected": "true_reputation_sum_expected_history",
    "active_actor_ids": "active_actor_ids_history",
    "active_participant_ids": "active_participant_ids_history",
    "observed_utility_matrix": "observed_utility_matrix_history",
    "eta_v": "eta_v_history",
    "gossip_target_ids": "gossip_target_ids_history",
    "averaging_agent_ids": "averaging_agent_ids_history",
    "avg_s_by_target": "avg_s_by_target_history",
    "delta_v_matrix": "delta_v_matrix_history",
}

# Fields that refresh_last_tracked_state rewrites (1320-1404). Everything else in
# StepRecord is written once, at append time, and never revised.
_REFRESHED = (
    "follower_counts",
    "paper_welfare_all_agents", "paper_welfare_followers_only",
    "status_count", "pu_count", "rep_count",
    "role_label",
    "estimated_reward_pu", "estimated_reward_rep", "estimated_reward_status",
    "actor_interaction_rate",
    "selected_reputation", "weighted_selected_reputation",
    "highest_rep_agent", "following",
    "dense_reputation", "dense_personal_benefit",
    "true_reputation", "true_reputation_rank",
    "true_reputation_theta", "true_reputation_sum_expected",
)


# ============================================================================
# Run results
# ============================================================================

@dataclass
class SimulationResults:
    """
    Column-major history. Field names match the original dict keys so the three
    sweep harnesses keep working during migration.

    Optional fields are appended only on steps where they were collected, so their
    histories can be SHORTER than the core ones. Use step_count(), never
    len(some_history), for the number of steps.
    """

    # --- core (both modes); NOT = skipped by overwrite_last ---
    follower_counts: list[list[int]] = field(default_factory=list)
    actor_counts: list[int] = field(default_factory=list)                       # NOT
    participant_counts: list[int] = field(default_factory=list)                 # NOT
    online_active_actor_payoff_sum: list[float] = field(default_factory=list)   # NOT
    paper_welfare_all_agents: list[float] = field(default_factory=list)
    paper_welfare_followers_only: list[float] = field(default_factory=list)
    social_welfare: list[float] = field(default_factory=list)  # alias of followers-only (2410)
    status_counts: list[int] = field(default_factory=list)
    pu_counts: list[int] = field(default_factory=list)
    rep_counts: list[int] = field(default_factory=list)
    role_label_history: list[list[str]] = field(default_factory=list)

    # --- ragged: one entry per role-update epoch, not per step (2413) ---
    role_update_times: list[int] = field(default_factory=list)                  # NOT

    # --- compact histories ---
    estimated_reward_pu_history: list[list[float]] = field(default_factory=list)
    estimated_reward_rep_history: list[list[float]] = field(default_factory=list)
    estimated_reward_status_history: list[list[float]] = field(default_factory=list)
    actor_interaction_rate_history: list[list[float]] = field(default_factory=list)
    selected_reputation_history: list[list[float]] = field(default_factory=list)
    weighted_selected_reputation_history: list[list[float]] = field(default_factory=list)
    highest_rep_agent_history: list[list[int]] = field(default_factory=list)
    following_history: list[list[int]] = field(default_factory=list)

    # --- FULL only ---
    norm_consensus: list[float] = field(default_factory=list)                   # NOT
    expected_utilities: list[dict[int, float]] = field(default_factory=list)    # NOT
    actor_rates: list[list[float]] = field(default_factory=list)                # NOT
    roles_history: list[list[AgentRole]] = field(default_factory=list)          # NOT
    actual_payoffs: list[dict[int, float]] = field(default_factory=list)        # NOT

    # --- dense, opt-in ---
    dense_reputation_history: list[np.ndarray] = field(default_factory=list)
    dense_personal_benefit_history: list[np.ndarray] = field(default_factory=list)
    true_reputation_history: list[np.ndarray] = field(default_factory=list)
    true_reputation_rank_history: list[np.ndarray] = field(default_factory=list)
    true_reputation_theta_history: list[np.ndarray] = field(default_factory=list)
    true_reputation_sum_expected_history: list[np.ndarray] = field(default_factory=list)
    active_actor_ids_history: list[list[int]] = field(default_factory=list)
    active_participant_ids_history: list[list[int]] = field(default_factory=list)
    observed_utility_matrix_history: list[np.ndarray] = field(default_factory=list)
    eta_v_history: list[float] = field(default_factory=list)
    gossip_target_ids_history: list[list[int]] = field(default_factory=list)
    averaging_agent_ids_history: list[list[int]] = field(default_factory=list)
    avg_s_by_target_history: list[dict[int, float]] = field(default_factory=list)
    delta_v_matrix_history: list[np.ndarray] = field(default_factory=list)

    # --- audit payloads, written by the recorder, not by append() ---
    role_update_diagnostics: list[dict] = field(default_factory=list)
    true_reputation_checkpoints: list[dict] = field(default_factory=list)
    estimate_consensus_checkpoints: list[dict] = field(default_factory=list)
    rate_audit_checkpoints: list[dict] = field(default_factory=list)

    # --- run summary, set once at the end (2536-2538) ---
    final_roles: Optional[list[AgentRole]] = None
    final_followers: Optional[list[int]] = None
    opinion_leader: int = -1

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def append(self, rec: StepRecord, *, role_updated: bool = False) -> None:
        """Append every non-None field of `rec`."""
        for src, dst in _FIELD_MAP.items():
            value = getattr(rec, src)
            if value is not None:
                getattr(self, dst).append(value)

        # social_welfare aliases the followers-only welfare (2410).
        self.social_welfare.append(rec.paper_welfare_followers_only)

        if role_updated:
            self.role_update_times.append(int(rec.t))

    def overwrite_last(self, rec: StepRecord) -> None:
        """
        Rewrite the most recent entry, for the subset of fields that
        refresh_last_tracked_state touches (1320-1404).

        Used by async harnesses that apply subset role updates after step()-level
        tracking, so timestep t reflects the post-update follower graph.

        No-op on an empty history, matching the guard at 1320-1321.
        """
        if not self.follower_counts:
            return

        for src in _REFRESHED:
            value = getattr(rec, src)
            if value is None:
                continue
            history = getattr(self, _FIELD_MAP[src])
            if history:
                history[-1] = value

        if self.social_welfare:
            self.social_welfare[-1] = rec.paper_welfare_followers_only

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def step_count(self) -> int:
        """Tracked steps. follower_counts is written every step in both modes,
        unlike norm_consensus (FULL only)."""
        return len(self.follower_counts)

    def __len__(self) -> int:
        return self.step_count()

    def validate(self) -> None:
        """
        Assert that every per-step history that was written at all has one entry
        per step. Catches a field appended under one condition and read under
        another — the failure mode the original's setdefault pattern invited.

        Ragged and summary fields are exempt. Call from tests.
        """
        exempt = {
            "role_update_times", "social_welfare",
            "role_update_diagnostics", "true_reputation_checkpoints",
            "estimate_consensus_checkpoints", "rate_audit_checkpoints",
            "final_roles", "final_followers", "opinion_leader",
        }
        n = self.step_count()
        for f in fields(self):
            if f.name in exempt:
                continue
            history = getattr(self, f.name)
            if not isinstance(history, list) or not history:
                continue
            if len(history) != n:
                raise AssertionError(
                    f"{f.name} has {len(history)} entries, expected {n}"
                )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def to_npz(self, path: str, config_dict: dict) -> None:
        """
        Save with the schema stamp and the full config, so a stale results file is
        detectable rather than inferred from a timestamp.

        Ragged and object-valued fields (per-agent lists, dicts, matrices) are
        stored with dtype=object, which requires allow_pickle=True on load. If you
        want pickle-free loading, restrict this to the rectangular fields your
        plots actually read and drop the rest.
        """
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "config": np.array(config_dict, dtype=object),
        }
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            try:
                arr = np.asarray(value)
                if arr.dtype == object:
                    raise ValueError
            except (ValueError, TypeError):
                arr = np.array(value, dtype=object)
            payload[f.name] = arr
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(cls, path: str) -> "SimulationResults":
        """
        Load and check the schema stamp. Raises if the file predates the current
        engine — regenerate rather than silently comparing incomparable runs.
        """
        data = np.load(path, allow_pickle=True)
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"results at schema {version}, engine at {SCHEMA_VERSION} — regenerate"
            )
        names = {f.name for f in fields(cls)}
        kwargs = {key: data[key].tolist() for key in data.files if key in names}
        return cls(**kwargs)