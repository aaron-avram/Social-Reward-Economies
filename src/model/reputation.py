"""
Reputation learning: Eq. (4) personal-benefit estimates and Eq. (9) observed reputation.

Design rules for this module:
  * The dense matrices are the ONLY representation of v and s. No per-agent dicts,
    no sync helpers.
  * Functions take and return state; they do not reach into Agent or MultiAgentSystem.
    The single exception is `Agent` import for AgentRole, which is not needed here at all
    (deliberately: reputation learning is role-independent).
  * Randomness enters through an explicit `rng` argument, used only for tie-breaking.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, List

import numpy as np
from numpy.random import Generator

from model.config import AlgorithmParams, Eq9Mode, LeaderUpdateMode

NO_LEADER = -1


# ============================================================================
# State
# ============================================================================

@dataclass
class ReputationState:
    """
    v: (N, N) personal-benefit estimates   v_i(k, t)   — Eq. (4)
    s: (N, N) observed reputation estimates s_i(k, t)  — Eq. (9)
    L: (N,)   highest-reputation target     L_i(t), NO_LEADER when unresolved
    """
    v: np.ndarray
    s: np.ndarray
    L: np.ndarray


    @classmethod
    def initial(cls, num_agents: int) -> "ReputationState":
        """
        Initialize Reputation State
        """
        return cls(
            v=np.zeros((num_agents, num_agents), dtype=float),
            s=np.zeros((num_agents, num_agents), dtype=float),
            L=np.full(num_agents, NO_LEADER, dtype=int),
        )


    @property
    def num_agents(self) -> int:
        """
        Number of Agents in State
        """
        return self.L.shape[0]


@dataclass
class Phase4Trace:
    """
    Diagnostic payload. Built only when a recorder asks for it — the `delta_v` copy
    is (N, N) per step and must not be allocated on the production path.

    Replaces the Dict[str, object] returned by _phase4_updates_numpy_fast (1612-1620).
    """
    gossip_target_ids: list[int] = field(default_factory=list)
    averaging_agent_ids: list[int] = field(default_factory=list)
    leader_update_agent_ids: list[int] = field(default_factory=list)
    avg_s_by_target: dict[int, float] = field(default_factory=dict)
    delta_v: Optional[np.ndarray] = None
    eq9_mode: Optional[Eq9Mode] = None
    leader_mode: Optional[LeaderUpdateMode] = None

# ============================================================================
# Leader selection — Section 6.4.4
# ============================================================================

def select_leader_from_row(
    s_row: np.ndarray,
    agent_id: int,
    delta: float,
    rng: Generator,
) -> int:
    """
    Highest-reputation target for one agent, over C \\ {i}, breaking near-ties
    (within `delta`) uniformly at random. Returns a Python int.
    """

    others = np.delete(np.arange(s_row.shape[0]), agent_id)
    if others.size == 0:                       # single-agent config
        return agent_id
    vals = s_row[others]
    return int(rng.choice(others[vals >= vals.max() - delta]))


def resolve_missing_leaders(
    state: ReputationState,
    agent_ids: Sequence[int],
    delta: float,
    rng: Generator,
) -> None:
    """
    Fill L[i] for any i in `agent_ids` still holding NO_LEADER. Mutates state.L.

    NOTE — this makes explicit a lazy write that is currently HIDDEN inside
    _compute_gossip_target_ids_from_active_participants (975-981): that function
    silently resolves L as a side effect of "computing" targets. Keep the resolution
    step separate so `gossip_targets` below is a pure read.
    """
    n = state.num_agents
    for i in agent_ids:
        i = int(i)
        if not (0 <= i < n):
            continue
        if state.L[i] != NO_LEADER:
            continue
        state.L[i] = select_leader_from_row(state.s[i, :], i, delta, rng)


def update_leaders(
    state: ReputationState,
    agent_ids: Sequence[int],
    delta: float,
    rng: Generator,
    *,
    source_s: Optional[np.ndarray] = None,
) -> None:
    """
    Unconditional L_i update for the given agents, reading `source_s` if supplied
    (the pre-Eq.(9) snapshot) and state.s otherwise. Mutates state.L.
    """
    s = state.s if source_s is None else source_s
    for i in agent_ids:
        i = int(i)
        state.L[i] = select_leader_from_row(s[i], i, delta, rng)


# ============================================================================
# Eq. (9) — observed reputation via gossip
# ============================================================================

def gossip_targets(state: ReputationState, participant_ids: np.ndarray) -> np.ndarray:
    """
    B(t) = union over active participants i of {L_i(t)}, excluding self-targets and
    out-of-range ids, sorted ascending.

    Precondition: every participant's L is resolved (call resolve_missing_leaders first).
    Pure read — must not mutate state.
    """
    n = state.num_agents

    ids = participant_ids[(participant_ids >= 0) & (participant_ids < n)]
    targets = state.L[ids]
    valid = (targets != NO_LEADER) & (targets >= 0) & (targets < n) & (targets != ids)
    return np.unique(targets[valid])


def eq9_averaging_ids(
    participant_ids: np.ndarray, num_agents: int, mode: Eq9Mode
) -> List:
    """
    Which agents' s-rows enter the average in Eq. (9).
    """
    if mode == Eq9Mode.PARTICIPANTS_ONLY:
        return [int(agent_id) for agent_id in participant_ids]
    return list(range(num_agents))


def leader_update_ids(
    participant_ids: np.ndarray, num_agents: int, mode: LeaderUpdateMode
) -> List[int]:
    """
    Which agents update L_i this step.
    """
    if mode == LeaderUpdateMode.ALL_AGENTS_POST_EQ9:
        return list(range(int(num_agents)))
    return [int(agent_id) for agent_id in participant_ids]


def apply_eq9(
    s: np.ndarray,
    delta_v: np.ndarray,
    participant_ids: np.ndarray,
    averaging_ids: np.ndarray,
    target_ids: np.ndarray,
    trace: bool = False
) -> tuple[np.ndarray, dict[int, float]]:
    """
    s_i(k, t+1) = mean_j s_j(k, t) + delta_v_i(k, t), for i in participants, k in B(t).
    Columns outside B(t) are untouched.

    Returns (s_new, avg_s_by_target). Build the dict only when tracing.
    """

    if target_ids.size == 0 or participant_ids.size == 0:
        return s, {}

    avg_s = np.mean(s[np.ix_(averaging_ids, target_ids)], axis=0)

    s_new = s.copy()
    s_new[np.ix_(participant_ids, target_ids)] = (avg_s[np.newaxis, :] + delta_v[np.ix_(participant_ids, target_ids)])

    avg_by_target = (
        {int(k): float(v) for k, v in zip(target_ids.tolist(), avg_s)} if trace else {}
    )
    return s_new, avg_by_target


# ============================================================================
# Orchestration
# ============================================================================

def phase4(
    state: ReputationState,
    observed_utility: np.ndarray,
    active_actor_ids: np.ndarray,
    active_participant_ids: np.ndarray,
    eta_v: float,
    params: AlgorithmParams,
    eq9_mode: Eq9Mode,
    leader_mode: LeaderUpdateMode,
    rng: Generator,
    *,
    update_leader_estimates: bool = True,
    trace: bool = False,
) -> tuple[ReputationState, Optional[Phase4Trace]]:
    """
    One Phase-4 step. Single implementation replacing both
    _phase4_updates_numpy_fast (1533-1620) and _phase4_updates_python (1622-1717).

    Order matters and must match the original exactly:
      1. snapshot s if leader_mode is PARTICIPANTS_ONLY_PRE_EQ9   (1554-1556)
      2. Eq. (4): v, delta_v                                       (1558-1564)
      3. if participants:
           a. resolve missing L, then B(t)                         (1567-1578)
           b. averaging set, Eq. (9)                               (1579-1594)
           c. update L from snapshot or post-Eq.(9) s              (1596-1606)

    The `update_actor_rates` loop at 1608-1610 is NOT here. Rate learning is not a
    reputation concern and calling Agent methods from this module would put a
    mutation of agent state behind a reputation-shaped signature. Move it to the
    Phase-4 caller in system.py.
    """

    prev_v = state.v
    pre_s_matrix = None

    if update_leader_estimates and leader_mode is LeaderUpdateMode.PARTICIPANTS_ONLY_PRE_EQ9:
        pre_s_matrix = np.array(state.s, dtype=float, copy=True)

    new_v = prev_v * (1.0 - eta_v)
    if active_actor_ids.size > 0:
        new_v[:, active_actor_ids] = (
            prev_v[:, active_actor_ids]
            + eta_v * (observed_utility[:, active_actor_ids] - prev_v[:, active_actor_ids])
        )
    delta_v = new_v - prev_v
    state.v = new_v

    gossip_target_ids = np.array([], dtype=int)
    avg_by_target: Dict[int, float] = {}
    averaging_agent_ids: np.ndarray[int] = np.array([], dtype=int)
    leader_update_agent_ids: np.ndarray[int] = np.array([], dtype=int)
    if active_participant_ids.size > 0:
        resolve_missing_leaders(state, active_participant_ids, params.delta, rng)
        gossip_target_ids = gossip_targets(state, active_participant_ids)
        averaging_agent_ids = eq9_averaging_ids(active_participant_ids, state.num_agents, eq9_mode)
        new_s, avg_by_target = apply_eq9(state.s, delta_v, active_participant_ids, averaging_agent_ids, gossip_target_ids, trace=trace)
        state.s = new_s

        if update_leader_estimates:
            leader_update_agent_ids = leader_update_ids(
                    active_participant_ids,
                    state.num_agents,
                    leader_mode
                )
            update_leaders(
                state,
                leader_update_agent_ids,
                params.delta,
                rng,
                source_s=pre_s_matrix if leader_mode is LeaderUpdateMode.PARTICIPANTS_ONLY_PRE_EQ9 else None,
)


    trace_obj = None
    if trace:
        trace_obj = Phase4Trace(
            gossip_target_ids=[int(x) for x in gossip_target_ids],
            averaging_agent_ids=[int(x) for x in averaging_agent_ids],
            leader_update_agent_ids=[int(x) for x in leader_update_agent_ids],
            avg_s_by_target=avg_by_target,
            delta_v=delta_v.copy(),
            eq9_mode=eq9_mode,
            leader_mode=leader_mode,
        )
    return state, trace_obj