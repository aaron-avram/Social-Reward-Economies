"""
Audit and diagnostic recording.

Replaces 476 lines spread across MultiAgentSystem — the largest single concern in
that class, larger than the simulation loop itself. Under NullRecorder none of it
runs, which is both the correctness win (no `if audit_rows is not None` branches
interleaved with role decisions) and the performance win for large-N sweeps.

Design notes:

  * The protocol is CHUNKY: one call per decision point, not per field. These fire
    at most once per agent per role-update epoch, never in an inner loop.
  * `role_update_decision` is FIRST-WRITE-WINS. That reproduces line 2368, where
    FALLBACK_TO_PU only overwrites the initial NOT_IN_C sentinel and leaves an
    earlier step-1 decision (e.g. STAY_PU_REP_BELOW_THRESHOLD) intact. Every other
    decision site in the original writes unconditionally, but each is the first
    write for its agent, so uniform first-write-wins is exactly equivalent.
  * The four enable_* methods and the four _*_enabled flags on MultiAgentSystem
    (594-599) collapse into WHICH recorder the caller constructs.
  * All enum values are unwrapped with .value and all numpy scalars cast to int/
    float, because these rows are exported as JSON/CSV. The original used
    str(self.config.eq9_averaging_mode) on a string field (1068); with enums that
    would produce "Eq9Mode.PARTICIPANTS_ONLY" instead of "participants_only".
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from model.agent import Agent, AgentRole
from model.config import AlgorithmParams, Eq9Mode, LeaderUpdateMode
from model.reputation import NO_LEADER, Phase4Trace
from model.welfare import TrueReputation, resolve_root_leader

NOT_IN_C = "NOT_IN_C"


# ============================================================================
# Protocol
# ============================================================================

class Recorder(Protocol):
    """
    Every method must be safe to call unconditionally — callers do NOT guard with
    `if recording:`. That is the point: it removes the ten `if audit_rows is not
    None` branches from _update_roles_sequential.
    """

    # --- role update (Section 7) ---
    def role_update_begin(self, t: int, state, signals, updatable: set[int],
                          params: AlgorithmParams) -> None: ...
    def role_update_step1(self, agent_id: int, **fields: Any) -> None: ...
    def role_update_decision(self, agent_id: int, code: str) -> None: ...
    def role_update_end(self, state, agents: Sequence[Agent]) -> None: ...

    # --- per-step ---
    def phase4(self, t: int, trace: Optional[Phase4Trace]) -> None: ...
    def rate_terms(self, t: int, agent_id: int, terms: dict) -> None: ...

    # --- does this recorder want the expensive payloads built at all? ---
    @property
    def wants_phase4_trace(self) -> bool: ...
    @property
    def wants_dense_history(self) -> bool: ...
    @property
    def wants_compact_history(self) -> bool: ...


class NullRecorder:
    """
    Production default. Every method is a no-op and both `wants_*` are False, so
    callers skip building traces entirely — the delta_v copy alone is N*N floats
    per step (320 KB at N=200).
    """

    wants_phase4_trace = False
    wants_dense_history = False
    wants_compact_history = False

    def role_update_begin(self, t, state, signals, updatable, params): pass
    def role_update_step1(self, agent_id, **fields): pass
    def role_update_decision(self, agent_id, code): pass
    def role_update_end(self, state, agents): pass
    def phase4(self, t, trace): pass
    def rate_terms(self, t, agent_id, terms): pass


# ============================================================================
# Full recorder
# ============================================================================

@dataclass
class FullRecorder:
    """
    Collects per-decision audit rows. Rows accumulate in memory — at N=200 over
    2000 steps with dense history on this is gigabytes, which is why it is opt-in.

    Replaces:
        enable_async_decision_audit         (775-782)  -> construct with async_audit=True
        enable_role_update_diagnostics      (791-799)  -> role_update_diagnostics=True
        enable_small_n_trace_export         (801-809)  -> dense_history=True
        set_decision_audit_preleader        (784-789)  -> preleader_id field
    """

    async_audit: bool = True
    role_update_diagnostics: bool = False
    dense_history: bool = False
    preleader_id: Optional[int] = None

    rows: list[dict] = field(default_factory=list)
    _pending: dict[int, dict] = field(default_factory=dict)
    _t: int = 0

    @property
    def wants_compact_history(self) -> bool:
        """The per-agent estimate/rate/leader histories.

        Both enable_async_decision_audit (781-782) and enable_small_n_trace_export
        (808-809) set _compact_debug_histories_enabled, so either flag implies it.
        Dense snapshots are recorded from inside the compact block (2484), so
        dense_history without compact would produce an inconsistent record.
        """
        return self.async_audit or self.dense_history

    @property
    def wants_phase4_trace(self) -> bool:
        return self.dense_history

    @property
    def wants_dense_history(self) -> bool:
        return self.dense_history

    # ------------------------------------------------------------------
    # role update
    # ------------------------------------------------------------------

    def role_update_begin(self, t, state, signals, updatable, params) -> None:
        """Seed one row per updatable agent. Body from 2136-2187."""
        self._t = int(t)
        self._pending = {}
        if not self.async_audit:
            return

        initial_leader_count = sum(1 for f in state.followers.values() if f)

        for i in sorted(updatable):
            sig = signals[i]
            target = int(sig.target)
            following_before = state.following[i]
            following_before = -1 if following_before is None else int(following_before)

            # 2149-2153: reputation toward the agent currently being followed.
            current_followed_rep = (
                float(sig.rep_row[following_before]) if following_before >= 0 else 0.0
            )
            # 2154-2158: reputation toward the pre-perturbation leader, NaN when unset.
            rep_to_preleader = (
                float(sig.rep_row[self.preleader_id])
                if self.preleader_id is not None else float("nan")
            )

            self._pending[i] = {
                "t": self._t,
                "agent_id": int(i),
                "scheduled_for_update": True,
                "current_role": sig.role.value,
                "has_followers": bool(state.followers[i]),
                "in_C": not state.followers[i],       # 2206-2208, folded in here
                "following_before": following_before,
                "highest_rep_agent_estimate": -1 if target == NO_LEADER else target,
                "selected_reputation_raw": float(sig.target_rep),
                "selected_reputation_weighted": float(params.gamma) * float(sig.target_rep),
                "rep_to_preleader_raw": rep_to_preleader,
                "current_followed_reputation_raw": current_followed_rep,
                "current_followed_reputation_weighted":
                    float(params.gamma) * current_followed_rep,
                "estimated_reward_pu": float(sig.estimated_reward_pu),
                "effective_threshold": None,
                "opinion_leader_count": int(initial_leader_count),
                "hysteresis_active": False,
                "step1_rep_signal_raw": 0.0,
                "step1_rep_signal_weighted": 0.0,
                "step1_condition_met": False,
                "best_k_before_redirect": -1,
                "best_k_after_redirect": -1,
                "redirect_applied": False,
                "redirect_target_is_follower": False,
                "new_role": sig.role.value,
                "following_after": following_before,
                "decision_code": NOT_IN_C,
            }

    def role_update_step1(self, agent_id: int, **fields: Any) -> None:
        """Field-level update; always overwrites. 2247-2252, 2259-2260, 2280-2284."""
        row = self._pending.get(agent_id)
        if row is not None:
            row.update(fields)

    def role_update_decision(self, agent_id: int, code: str) -> None:
        """
        FIRST WRITE WINS — see the note at the top of this file. This is what makes
        step 3's FALLBACK_TO_PU leave an earlier step-1 decision alone (2368).
        """
        row = self._pending.get(agent_id)
        if row is None:
            return                                  # not scheduled this epoch
        if row["decision_code"] != NOT_IN_C:
            return                                  # already decided
        row["decision_code"] = str(code)

    def role_update_end(self, state, agents: Sequence[Agent]) -> None:
        """Fill final role/following and flush. Body from 2375-2380."""
        for i, row in self._pending.items():
            row["new_role"] = agents[i].state.role.value
            following = agents[i].state.following
            row["following_after"] = -1 if following is None else int(following)
            self.rows.append(row)
        self._pending = {}

    # ------------------------------------------------------------------
    # per-step
    # ------------------------------------------------------------------

    def phase4(self, t: int, trace: Optional[Phase4Trace]) -> None:
        """Retained for the dense-history path; StepRecord reads the trace directly."""
        pass

    def rate_terms(self, t: int, agent_id: int, terms: dict) -> None:
        pass


# ============================================================================
# Checkpoint row builders — called on demand at role-update epochs, not per step
# ============================================================================

def _mode_fields(eq9_mode: Eq9Mode, leader_mode: LeaderUpdateMode) -> dict:
    """The two mode columns every checkpoint row carries (1068-1069, 1136-1137,
    1182-1183). .value, not str(), or the enum name leaks into the export."""
    return {
        "eq9_averaging_mode": eq9_mode.value,
        "leader_update_mode": leader_mode.value,
    }


def true_reputation_rows(
    t: int, tr: TrueReputation, num_agents: int,
    *, checkpoint_kind: str, role_update_index: int,
    eq9_mode: Eq9Mode, leader_mode: LeaderUpdateMode,
) -> list[dict]:
    """One row per agent. Body from _build_true_reputation_checkpoint_rows (1034-1072).

    The _sync_reputation_views_for_diagnostics() call at 1040 is GONE — there is one
    reputation matrix now, so there is nothing to sync.
    """
    n_exact = int(np.sum(tr.exact_top_mask))
    n_near = int(np.sum(tr.near_top_mask))
    modes = _mode_fields(eq9_mode, leader_mode)

    return [
        {
            "t": int(t),
            "checkpoint_kind": str(checkpoint_kind),
            "role_update_index": int(role_update_index),
            "agent_id": int(i),
            "true_reputation": float(tr.true_reputation[i]),
            "true_rank": int(tr.true_rank[i]),
            "theta_mu": float(tr.theta_mu[i]),
            "sum_expected_utility_others": float(tr.sum_expected_utility_others[i]),
            "top_true_reputation": float(tr.top_value),
            "gap_to_true_top": float(tr.top_value - tr.true_reputation[i]),
            "true_top_unique": int(tr.unique_true_top_agent >= 0),
            "unique_true_top_agent": int(tr.unique_true_top_agent),
            "is_exact_top_tie": int(bool(tr.exact_top_mask[i]) and n_exact > 1),
            "is_near_top_tie": int(bool(tr.near_top_mask[i]) and n_near > 1),
            **modes,
        }
        for i in range(num_agents)
    ]


def estimate_consensus_rows(
    t: int, agents: Sequence[Agent], s: np.ndarray, L: np.ndarray,
    tr: TrueReputation, params: AlgorithmParams,
    *, checkpoint_kind: str, role_update_index: int,
    eq9_mode: Eq9Mode, leader_mode: LeaderUpdateMode,
) -> list[dict]:
    """
    One row per observer, comparing its reputation estimates against the true
    ranking. Body from _build_estimate_consensus_checkpoint_rows (1074-1140).

    Reads s[i, :] rather than agent.state.reputation_estimates — the dicts are gone.
    Ranking is by (-value, id), matching 1090.
    """
    num_agents = len(agents)
    modes = _mode_fields(eq9_mode, leader_mode)
    following = [a.state.following for a in agents]
    follower_counts = [len(a.state.followers) for a in agents]
    rows: list[dict] = []

    for i, agent in enumerate(agents):
        others = [k for k in range(num_agents) if k != i]
        others.sort(key=lambda k: (-float(s[i, k]), k))

        if others:
            top_agent = int(others[0])
            top_value = float(s[i, top_agent])
            if len(others) > 1:
                second_agent = int(others[1])
                second_value = float(s[i, second_agent])
            else:
                second_agent = -1
                second_value = top_value          # 1096: falls back to the top value
            candidate_count = int(
                sum(1 for k in others if float(s[i, k]) >= top_value - float(params.delta))
            )
        else:
            top_agent, top_value = -1, 0.0
            second_agent, second_value = -1, 0.0
            candidate_count = 0

        target = int(L[i])
        selected_target = -1 if target == NO_LEADER else target
        selected_value = float(s[i, selected_target]) if selected_target >= 0 else 0.0
        follows = -1 if following[i] is None else int(following[i])
        root = resolve_root_leader(i, following, follower_counts)
        has_followers = int(follower_counts[i] > 0)
        unique_top = int(tr.unique_true_top_agent)

        rows.append({
            "t": int(t),
            "checkpoint_kind": str(checkpoint_kind),
            "role_update_index": int(role_update_index),
            "observer_id": int(i),
            "role": agent.state.role.value,
            "has_followers": has_followers,
            "eligible_in_C": int(not has_followers),
            "highest_rep_agent_estimate": selected_target,
            "selected_rep_value": selected_value,
            "top_estimate_agent": top_agent,
            "top_estimate_value": top_value,
            "second_estimate_agent": second_agent,
            "second_estimate_value": second_value,
            "gap_top2": float(top_value - second_value),
            "candidate_count_within_delta": candidate_count,
            "true_top_unique": int(unique_top >= 0),
            "unique_true_top_agent": unique_top,
            "selected_matches_true_top": int(unique_top >= 0 and selected_target == unique_top),
            "following": follows,
            "current_root_leader": int(root),
            "current_root_matches_true_top": int(unique_top >= 0 and root == unique_top),
            **modes,
        })
    return rows


def rate_audit_rows(
    t: int, agents: Sequence[Agent],
    *, checkpoint_kind: str, role_update_index: int,
    eq9_mode: Eq9Mode, leader_mode: LeaderUpdateMode,
) -> list[dict]:
    """
    One row per agent comparing the paper's Eq. (13) driver against what the code
    actually used. Body from _build_rate_audit_checkpoint_rows (1142-1186).

    This is the ONLY caller of Agent.actor_rate_terms() — the production path uses
    actor_rate_driver(), which returns a scalar and skips the dict entirely.
    """
    modes = _mode_fields(eq9_mode, leader_mode)
    rows: list[dict] = []

    for i, agent in enumerate(agents):
        terms = agent.actor_rate_terms()
        paper_driver = float(terms["standard_driver"])
        code_driver = float(terms["driver"])
        st = agent.state

        rows.append({
            "t": int(t),
            "checkpoint_kind": str(checkpoint_kind),
            "role_update_index": int(role_update_index),
            "agent_id": int(i),
            "role": st.role.value,
            "follower_count": int(len(st.followers)),
            "estimated_reward_pu": float(st.estimated_reward_pu),
            "estimated_reward_rep": float(st.estimated_reward_rep),
            "estimated_reward_status": float(st.estimated_reward_status),
            "actor_interaction_rate": float(st.actor_interaction_rate),
            "participant_interaction_rate": float(st.participant_interaction_rate),
            "actor_rate_driver_mode": str(terms["actor_rate_driver_mode"]),
            "actor_rate_status_override_min_followers":
                int(terms["actor_rate_status_override_min_followers"]),
            "status_override_active": int(terms["status_override_active"]),
            "paper_driver_pu_term": float(terms["pu_term"]),
            "paper_driver_rep_term": float(terms["rep_term"]),
            "paper_driver_status_term": float(terms["status_term"]),
            "paper_driver_value": paper_driver,
            "paper_driver_label": str(terms["standard_driver_label"]),
            "code_driver_value": code_driver,
            "code_driver_label": str(terms["driver_label"]),
            "paper_driver_matches_code":
                int(np.isclose(paper_driver, code_driver, atol=1e-12, rtol=0.0)),
            "paper_section66_requires_status_update":
                int(st.role is AgentRole.PERSONAL_UTILITY and len(st.followers) > 0),
            "status_estimate_positive": int(float(st.estimated_reward_status) > 0.0),
            **modes,
        })
    return rows


def checkpoint_bundle(
    t: int, agents: Sequence[Agent], s: np.ndarray, L: np.ndarray,
    tr: TrueReputation, params: AlgorithmParams,
    *, checkpoint_kind: str, role_update_index: int,
    eq9_mode: Eq9Mode, leader_mode: LeaderUpdateMode,
) -> dict[str, list[dict]]:
    """All three checkpoint row sets. Body from build_expb_checkpoint_audit_bundle
    (1188-1207). `tr` is computed ONCE by the caller and shared — the original
    recomputed _compute_true_reputation_vector separately at 1041 and 1081."""
    kw = dict(checkpoint_kind=checkpoint_kind, role_update_index=role_update_index,
              eq9_mode=eq9_mode, leader_mode=leader_mode)
    return {
        "true_reputation_checkpoints": true_reputation_rows(t, tr, len(agents), **kw),
        "estimate_consensus_checkpoints":
            estimate_consensus_rows(t, agents, s, L, tr, params, **kw),
        "rate_audit_checkpoints": rate_audit_rows(t, agents, **kw),
    }


def role_update_diagnostic_row(
    t: int, agents: Sequence[Agent], s: np.ndarray, L: np.ndarray,
    params: AlgorithmParams, role_update_index: int,
) -> dict:
    """
    One aggregate row per role-update epoch: leader concentration, role counts, and
    the margin distributions that distinguish weak-following from fragmented-
    following. Body from _build_role_update_diagnostic_row (1209-1312).
    """
    num_agents = len(agents)
    follower_counts = [len(a.state.followers) for a in agents]
    ranked = sorted(range(num_agents), key=lambda i: (-follower_counts[i], i))

    def nth_leader(rank: int) -> tuple[int, int]:
        if rank >= len(ranked):
            return -1, 0
        leader = int(ranked[rank])
        count = int(follower_counts[leader])
        return (leader, count) if count > 0 else (-1, 0)

    top_id, top_n = nth_leader(0)
    second_id, second_n = nth_leader(1)
    third_id, third_n = nth_leader(2)

    highest_targets: list[int] = []
    following_targets: list[int] = []
    pu_estimates: list[float] = []
    rep_weighted_all: list[float] = []
    step1_margins: list[float] = []
    gate_margins: list[float] = []
    role_counts = {r.value: 0 for r in AgentRole}

    for i, agent in enumerate(agents):
        st = agent.state
        role_counts[st.role.value] += 1
        pu_est = float(st.estimated_reward_pu)
        pu_estimates.append(pu_est)

        target = int(L[i])
        if target != NO_LEADER:
            highest_targets.append(target)
            rep_raw = float(s[i, target])
        else:
            rep_raw = 0.0

        rep_weighted = float(params.gamma) * rep_raw
        rep_weighted_all.append(rep_weighted)
        step1_margins.append(rep_weighted - pu_est)

        # Same hysteresis rule as step 1 (1256-1263).
        followerless_reputation = (
            st.role is AgentRole.REPUTATION and follower_counts[i] == 0
        )
        threshold = float(params.B_F if followerless_reputation else params.B_R)
        gate_margins.append(rep_weighted - max(threshold, pu_est))

        if st.following is not None:
            following_targets.append(int(st.following))

    highest_counts = Counter(highest_targets)
    following_counts = Counter(following_targets)
    top_target_id, top_share, second_share = -1, 0.0, 0.0
    if highest_counts:
        top2 = highest_counts.most_common(2)
        denom = max(1, len(highest_targets))
        top_target_id = int(top2[0][0])
        top_share = float(top2[0][1] / denom)
        if len(top2) > 1:
            second_share = float(top2[1][1] / denom)

    denom_followers = max(1, num_agents - 1)

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def share_positive(xs: list[float]) -> float:
        return float(np.mean(np.asarray(xs, dtype=float) > 0.0)) if xs else 0.0

    return {
        "t": int(t),
        "role_update_index": int(role_update_index),
        "top_leader_id": top_id,
        "top_followers": top_n,
        "second_leader_id": second_id,
        "second_followers": second_n,
        "third_leader_id": third_id,
        "third_followers": third_n,
        "top_follower_share": float(top_n / denom_followers),
        "top2_follower_share": float((top_n + second_n) / denom_followers),
        "distinct_follow_targets": int(len(following_counts)),
        "n_reputation": int(role_counts[AgentRole.REPUTATION.value]),
        "n_personal_utility": int(role_counts[AgentRole.PERSONAL_UTILITY.value]),
        "n_status": int(role_counts[AgentRole.STATUS.value]),
        "mean_pu_estimate": mean(pu_estimates),
        "mean_rep_signal_weighted": mean(rep_weighted_all),
        "mean_step1_margin": mean(step1_margins),
        "share_step1_margin_positive": share_positive(step1_margins),
        "mean_gate_margin": mean(gate_margins),
        "share_gate_margin_positive": share_positive(gate_margins),
        "distinct_highest_rep_targets": int(len(highest_counts)),
        "top_highest_rep_target_id": top_target_id,
        "top_highest_rep_target_share": top_share,
        "second_highest_rep_target_share": second_share,
    }