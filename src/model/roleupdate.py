"""
Section 7: sequential three-step role update.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import math

import numpy as np
from numpy.random import Generator

from model.agent import Agent, AgentRole
from model.config import AlgorithmParams, ScheduleParams
from model.instrumentation import Recorder, NullRecorder


# ============================================================================
# Working state
# ============================================================================


@dataclass
class RoleUpdateState:
    """
    Mutable working copy for one invocation. P/R/S partition the agents by role;
    `followers[i]` is the set following i.

    INVARIANT (checked in `validate`, cheap enough to keep on in tests):
      * P, R, S are pairwise disjoint and cover range(N)
      * j in followers[i]  <=>  agents[j].following == i, after `apply`
      * no agent follows itself
      * no chains: if i in R then followers[i] is empty
    """
    P: set[int]
    R: set[int]
    S: set[int]
    followers: dict[int, set[int]]
    following: dict[int, Optional[int]]

    @classmethod
    def from_agents(cls, agents: Sequence[Agent]) -> "RoleUpdateState":
        """
        From agents
        """
                
        # Initialize: copy current state
        P = set(i for i, a in enumerate(agents) if a.state.role == AgentRole.PERSONAL_UTILITY)
        R = set(i for i, a in enumerate(agents) if a.state.role == AgentRole.REPUTATION)
        S = set(i for i, a in enumerate(agents) if a.state.role == AgentRole.STATUS)
        
        # Maintain follower relationships during update
        followers = {i: set(agents[i].state.followers) for i in range(len(agents))}
        following = {i : agents[i].state.following for i in range(len(agents))}

        return cls(P, R, S, followers, following)

    def apply(self, agents: Sequence[Agent]) -> None:
        """
        Apply
        """
        for i, agent in enumerate(agents):
            agent.state.followers = self.followers[i]
            agent.state.following = self.following[i]
            agent.state.role = (
                AgentRole.PERSONAL_UTILITY if i in self.P
                else AgentRole.REPUTATION if i in self.R
                else AgentRole.STATUS
            )

    def detach(self, agent_id: int) -> None:
        """Remove agent_id from every follower set (defensive: the paper removes it
        from one set, this keeps the graph consistent if stale membership exists).
        """
        for i, follower_set in self.followers.items():
            if agent_id in follower_set:
                self.followers[i].remove(agent_id)


    def validate(self, num_agents: int) -> None:
        """Assert the invariants above. Call in tests, not on the hot path."""
        assert len(self.P) + len(self.R) + len(self.S) == num_agents
        assert set.isdisjoint(self.P, self.R) and set.isdisjoint(self.R, self.S) and set.isdisjoint(self.P, self.S)


def resolve_updatable(update_candidates: Optional[Iterable[int]], num_agents: int) -> set[int]:
    """
    Agents allowed to reevaluate this call. None => all agents (synchronous mode).
    Out-of-range ids are dropped; an empty result means the caller should return early.
    """
    if update_candidates is None:
        return set(range(num_agents))
    else:
        return {int(i) for i in update_candidates if 0 <= int(i) < num_agents}
    


# ============================================================================
# Step 1 — Reputation (Section 7.3)
# ============================================================================

def _effective_threshold(agent_id: int, C_r: set[int], params: AlgorithmParams) -> tuple[float, bool]:
    """
    Hysteresis (Section 7.1.3): B_F for agents already following, B_R otherwise.
    Returns (B_i, hysteresis_active).
    """
    hysteresis_active = (agent_id in C_r and params.B_F < params.B_R)
    return (params.B_F if hysteresis_active else params.B_R, hysteresis_active)


def _redirect_target(
    best_k: Optional[int],
    state: RoleUpdateState,
) -> tuple[Optional[int], bool]:
    """
    [ROLE-3] If the chosen target is itself a follower, follow its leader instead,
    to avoid indirect chains. Returns (best_k, target_was_follower).
    """
    leader = state.following[best_k]
    if leader is None:
        return best_k, False
    return leader, True


def step1_reputation(
    state: RoleUpdateState,
    signals: dict[int, "AgentSignals"],
    updatable: set[int],
    num_agents: int,
    params: AlgorithmParams,
    rng: Generator,
    rec: Recorder = NullRecorder(),
) -> None:
    """
    Agents decide whether to follow their highest-reputation target.

    Condition [ROLE-2], Section 7.3:   γ · s_i(L_i, t)  >  max(B_i, Ĵ^pu_i)
    """
    C = set(i for i in range(num_agents) if len(state.followers[i]) == 0)
    C_r = C & state.R

    update_order = list(C & updatable)
    rng.shuffle(update_order)

    for i in update_order:
        sig = signals[i]
        B_i, hysteresis_active = _effective_threshold(i, C_r, params)
        rep_weighted = params.gamma * sig.target_rep
        step1_condition = rep_weighted > max(B_i, signals[i].estimated_reward_pu)

        rec.role_update_step1(
            i,
            effective_threshold=B_i,
            hysteresis_active=hysteresis_active,
            step1_rep_signal_raw=sig.target_rep,
            step1_rep_signal_weighted=rep_weighted,
            step1_condition_met=step1_condition
        )

        if not step1_condition:
            rec.role_update_decision(i, "STAY_PU_REP_BELOW_THRESHOLD")
            if i in C_r:
                state.detach(i)
                state.following[i] = None
                state.R.discard(i)
            continue

        best_k = signals[i].target
        best_k, target_was_follower = _redirect_target(best_k, state)

        if best_k is None:
            rec.role_update_decision(i, "NO_TARGET")
            continue

        if best_k == i:
            rec.role_update_decision(i, "SELF_REDIRECT_BLOCK")
            continue

        if state.followers[i]:
            for follower_id in list(state.followers[i]):
                state.followers[best_k].add(follower_id)
                state.following[follower_id] = best_k
            state.followers[i].clear()

        state.detach(i)
        state.following[i] = best_k
        state.followers[best_k].add(i)
        state.R.add(i)
        state.P.discard(i)

        rec.role_update_decision(i, "FOLLOW_REDIRECT" if target_was_follower else "FOLLOW_DIRECT")
    print(state.following)

# ============================================================================
# Step 2 — Status (Section 7.4)
# ============================================================================

def status_threshold(c: float, num_agents: int) -> int:
    """|F_i| >= cN, so the smallest qualifying integer count is ceil(cN)."""
    return int(math.ceil(c * num_agents))


def step2_status(
    state: RoleUpdateState,
    signals: dict[int, "AgentSignals"],
    updatable: set[int],
    params: AlgorithmParams,
    num_agents: int,
    rec: Recorder = NullRecorder(),
) -> None:
    """
    Agents with >= ceil(cN) followers take STATUS if  κ · Ĵ^s_i  >  Ĵ^pu_i.

    [STATUS-1] depends on estimated_reward_status having been updated in Phase 3
    of the same timestep, for follower-holding actors.

    Iterates `updatable` (a set) — insertion order is not sorted. No RNG is
    consumed here, and decisions are independent, so the order does not affect
    the outcome. Sorting it is safe and worth doing for reproducible audit rows.
    """
    min_followers = status_threshold(params.c_threshold, num_agents)
    updatable = sorted(updatable)
    for i in updatable:
        if len(state.followers[i]) >= min_followers:
            if params.kappa * signals[i].estimated_reward_status > signals[i].estimated_reward_pu:
                if state.following[i] is not None:
                    state.detach(i)
                    state.following[i] = None

                state.S.add(i)
                state.P.discard(i)
                state.R.discard(i)

        if len(state.followers[i]) < min_followers:
            continue
        sig = signals[i]
        if params.kappa * sig.estimated_reward_status <= sig.estimated_reward_pu:
            continue
        if state.following[i] is not None:
            state.detach(i)
            state.following[i] = None
        state.S.add(i)
        state.P.discard(i)
        state.R.discard(i)
        rec.role_update_decision(i, "STATUS_TAKEN") 


# ============================================================================
# Step 3 — Personal utility fallback (Section 7.5)
# ============================================================================

def step3_fallback_pu(
    state: RoleUpdateState,
    updatable: set[int],
    rec: Recorder = NullRecorder(),
) -> None:
    """
    Everything not in R or S falls back to PU, dropping any leader.
    """
    updatable = sorted(updatable)
    for i in updatable:
        if i not in state.R and i not in state.S:
            if state.following[i] is not None:
                state.detach(i)
                state.following[i] = None
            state.P.add(i)
            rec.role_update_decision(i, "FALLBACK_TO_PU")


# ============================================================================
# Signals — the read-only slice of agent state that steps 1-2 need
# ============================================================================

@dataclass(frozen=True)
class AgentSignals:
    """
    Pulled once per agent before the procedure starts, so the steps never reach
    into Agent. Keeps this module testable with plain data.
    """
    role: AgentRole
    target: int                  # L_i(t), NO_LEADER if unresolved
    target_rep: float            # s_i(L_i, t)
    estimated_reward_pu: float
    estimated_reward_status: float
    rep_row: np.ndarray


def collect_signals(
    agents: Sequence[Agent],
    rep_s: np.ndarray,
    rep_L: np.ndarray,
) -> dict[int, AgentSignals]:
    """
    Read s_i(L_i,t) straight from the reputation matrix rather than from
    agent.state.reputation_estimates.
    """
    return {
        i : AgentSignals(
            role=a.state.role,
            target=int(rep_L[i]),
            target_rep=(0.0 if rep_L[i] == -1
                        else float(rep_s[i, rep_L[i]])),
            estimated_reward_pu=float(a.state.estimated_reward_pu),
            estimated_reward_status=float(a.state.estimated_reward_status),
            rep_row=rep_s[i, :]
        )
        for i , a in enumerate(agents)
    }


# ============================================================================
# Orchestration
# ============================================================================

def update_roles_sequential(
    agents: Sequence[Agent],
    rep_s: np.ndarray,
    rep_L: np.ndarray,
    params: AlgorithmParams,
    rng: Generator,
    *,
    t: int = 0,
    update_candidates: Optional[Iterable[int]] = None,
    rec: Recorder = NullRecorder(),
) -> None:
    """
    Section 7 in full.

    Step 5 must come before step 1: it clears stale STATUS so zero-follower status
    agents do not persist.
    """
    num_agents = len(agents)
    updatable = resolve_updatable(update_candidates, num_agents)
    if not updatable:
        return
    
    state = RoleUpdateState.from_agents(agents)
    signals = collect_signals(agents, rep_s, rep_L)

    rec.role_update_begin(t, state, signals, updatable, params)
    
    _ = [state.S.discard(i) for i in updatable]

    step1_reputation(state, signals, updatable, num_agents, params, rng, rec)
    step2_status(state, signals, updatable, params, num_agents, rec)
    step3_fallback_pu(state, updatable, rec)

    state.apply(agents)
    rec.role_update_end(state, agents)


# ============================================================================
# Epoch schedule — Section 7.1.4
# ============================================================================

class RoleUpdateSchedule:
    """
    Update epochs s_n = s_{n-1} + T_n. Three input forms, in precedence order:
      1. explicit s_n list        (role_update_epochs)
      2. explicit T_n sequence    (role_update_T_sequence)
      3. generated: constant T    (fixed_role_update_interval)
         or geometrically increasing from role_update_base_interval
    """

    def __init__(self, sched: ScheduleParams):
        t_seq = [int(t) for t in sched.role_update_T_sequence if int(t) > 0]
        if t_seq:
            s_prev = max(0, int(sched.role_update_s0))
            epochs = []
            for t_n in t_seq:
                s_prev += t_n
                if s_prev > 0:
                    epochs.append(int(s_prev))
            self.epochs = sorted(set(epochs))
        else:
            self.epochs = sorted(set(int(t) for t in sched.role_update_epochs if int(t) > 0))
        self.explicit = bool(self.epochs)
        self.base = max(1, int(sched.role_update_base_interval))
        self.fixed = bool(sched.fixed_role_update_interval)

        self._next_time = self.epochs[0] if self.explicit else self.base

        self.n = 0

    def _interval(self, k: int) -> int:
        """T_n after n completed epochs """
        return self.base if self.fixed else max(self.base, int(self.base * (1.0 + k * 0.1)))

    def _num_epochs(self) -> int:
        """Explicit lists are finite; generated schedules never exhaust."""
        return len(self.epochs) if self.explicit else 1 << 62

    def next_epoch(self) -> int:
        """Time of the n-th update epoch."""
        if self.explicit:
            return self.epochs[self.n]
        return sum(self._interval(k) for k in range(self.n+1))

    def due_count(self, t: int) -> int:
        """
        Epochs firing at time t. Advances internal state, so call exactly once
        per step.
        """
        fired = 0
        while self.n < self._num_epochs() and t >= self._next_time:
            self.n += 1
            self._next_time = (self.epochs[self.n] if self.explicit and self.n < len(self.epochs)
                               else self._next_time + self._interval(self.n))
            fired += 1
        return fired