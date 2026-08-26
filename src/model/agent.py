"""
Single Agent Simulation Functionality
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional

import numpy as np
from numpy.random import Generator

from model.config import AlgorithmParams, Dimensions, ActorRateDriverMode


class AgentRole(Enum):
    """
    Agent Roles during Simulation
    """
    PERSONAL_UTILITY = "personal_utility"
    REPUTATION = "reputation"
    STATUS = "status"


@dataclass
class AgentState:
    """Complete agent state for Sections 6–7"""

    # --- Policy weights ---
    weights_pu: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    weights_status: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # --- Role and follower relationships ---
    role: AgentRole = AgentRole.PERSONAL_UTILITY
    following: Optional[int] = None  # Which agent this agent follows (for REPUTATION role)
    followers: Set[int] = field(default_factory=set)  # Who follows this agent

    # --- Section 6.4: Reputation Learning ---
    # v_i(k,t): Personal benefit estimates (how much benefit agent i gets from agent k's behavior)
    personal_benefit_estimates: Dict[int, float] = field(default_factory=dict)

    # s_i(k,t): Observed reputation estimates (what agent i believes agent k's reputation is)
    reputation_estimates: Dict[int, float] = field(default_factory=dict)

    # L_i(t): Estimate of which agent has highest reputation
    highest_rep_agent_estimate: Optional[int] = None

    # --- Section 6.6: Reward Estimates (three separate) ---
    estimated_reward_pu: float = 0.0      # Ĵ^{pu}_i(t)
    estimated_reward_rep: float = 0.0     # Ĵ^r_i(t)
    estimated_reward_status: float = 0.0  # Ĵ^s_i(t)

    # --- Section 6.7: Actor Interaction Rates ---
    actor_interaction_rate: float = 0.7   # μ_{a,i}(t) - learned
    participant_interaction_rate: float = 0.7  # μ_{p,i} - fixed

    # Tracking for history
    payoff_history: List[float] = field(default_factory=list)

    # Flag to track if agent was following before (for hysteresis)
    was_following: bool = False


class Agent:
    """Agent implementing Sections 6–7 algorithms"""

    def __init__(self, agent_id: int, params: AlgorithmParams, dims: Dimensions, rng: Generator):
        self.agent_id = agent_id
        self.params = params
        self.dims = dims
        self.rng = rng

        self.state = AgentState()

        # Ensure policy parameter shapes follow runtime config (num_states, num_actions).
        # AgentState defaults are placeholders and may not match experiment overrides.
        self.state.weights_pu = self.rng.standard_normal(self.dims.num_states, self.dims.num_actions) * 0.1
        self.state.weights_status = self.rng.standard_normal(self.dims.num_states, self.dims.num_actions) * 0.1

        # Initialize reputation and personal benefit estimates for all agents
        for other_id in range(self.dims.num_agents):
            self.state.personal_benefit_estimates[other_id] = 0.0
            self.state.reputation_estimates[other_id] = 0.0

        # Runtime tuning: initial rates are configurable for faster large-scale experiments.
        self.state.actor_interaction_rate = float(self.params.initial_actor_interaction_rate)
        self.state.participant_interaction_rate = float(self.params.initial_participant_interaction_rate)

        # Initial highest-reputation target is resolved lazily from s_i(k,t) when
        # first needed so tied rows use the normal Section 6.4.4 tie rule over C\{i}.
        self.state.highest_rep_agent_estimate = None

        # Track last action for gradient updates
        self.last_action = None
        self.last_state = None

    # ==================== Section 6.3: Personal Utility ====================

    def get_softmax_policy(self, state: int, weights: np.ndarray) -> np.ndarray:
        """Convert policy weights to softmax probabilities"""
        logits = weights[state, :]
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-8)


    def get_behavior_weights(self) -> np.ndarray:
        """Return role-consistent behavior weights w_i(t) used for imitation."""
        if self.state.role == AgentRole.STATUS:
            return self.state.weights_status
        return self.state.weights_pu


    def get_current_policy(self, state: int, leader_weights: np.ndarray) -> np.ndarray:
        """
        Return the actual action distribution used by this agent at the current
        timestep under its present role and follow relationship.
        """
        if self.state.role == AgentRole.REPUTATION and self.state.following is not None:
            return self.get_softmax_policy(state, leader_weights)
        if self.state.role == AgentRole.STATUS:
            return self.get_softmax_policy(state, self.state.weights_status)
        return self.get_softmax_policy(state, self.state.weights_pu)


    def select_action(self, state: int, uniform: float, leader_weights: np.ndarray) -> int:
        """Select action based on current role"""
        self.last_state = state
        policy = self.get_current_policy(state, leader_weights)

        cdf = np.cumsum(policy)
        cdf /= cdf[-1]

        action = int(np.searchsorted(cdf, uniform, side="right"))
        self.last_action = action
        return action


    def update_policy_gradient(self, state: int, action: int, reward: float, 
                               weights: np.ndarray, lr: float) -> np.ndarray:
        """
        Policy gradient update (Eq. 8 for PU, Eq. 12 for status)
        w(t+1) = w(t) + α(t) · u_i(s,x) · ∇_w log π(w; s, x)
        where ∇_w log π = a - π(w; s, ·)
        """
        policy = self.get_softmax_policy(state, weights)
        action_one_hot = np.zeros(self.dims.num_actions)
        action_one_hot[action] = 1
        gradient = action_one_hot - policy

        weights_new = weights.copy()
        weights_new[state, :] += lr * gradient * reward
        return weights_new


    def update_personal_utility(self, state: int, action: int, reward: float,
                                alpha_pu_t: float, eta_J_t: float):
        """
        Section 6.3: Personal utility optimization
        - Update policy weights via gradient (Eq. 8)
        - Update reward estimate Ĵ^{pu}_i(t)
        """
        self.state.weights_pu = self.update_policy_gradient(
            state, action, reward, self.state.weights_pu, alpha_pu_t
        )

        # Update personal utility reward estimate (Eq. in Section 6.6)
        self.state.estimated_reward_pu += eta_J_t * (reward - self.state.estimated_reward_pu)

    # ==================== Section 6.5: Status Optimization ====================


    def update_status_optimization(self, state: int, action: int, 
                                   social_support_sum: float, beta_status_t: float,
                                   eta_J_t: float):
        """
        Section 6.5: Status optimization
        - Update policy weights via gradient (Eq. 12)
        - Update reward estimate Ĵ^s_i(t)
        
        NOTE: social_support_sum is the SUM of follower payoffs (Eq. 11),
        not the average. This is critical for convergence.
        """
        self.state.weights_status = self.update_policy_gradient(
            state, action, social_support_sum, self.state.weights_status, beta_status_t
        )

        # Update status reward estimate
        self.state.estimated_reward_status += eta_J_t * \
            (social_support_sum - self.state.estimated_reward_status)

    # ==================== Section 6.4.5: Reputation Optimization ====================

    def adopt_behavior(self, leader_weights: List = None):
        """
        Section 6.4.5: Agent following a leader directly copies the leader's behavior
        """
        if self.state.following is None or self.state.following == self.agent_id:
            return
        # [REP-5] Copy role-consistent leader behavior w_k(t), not always w_k^pu.
        self.state.weights_pu = np.copy(leader_weights)


    def update_reputation_reward_estimate(self, followed_rep_estimate: float):
        """
        [REP-6] Section 6.6:
        For active reputation optimizers, J^r_i(t) is the current reputation
        estimate of followed agent k, i.e., s_i(k,t), not an EMA of leader payoff.
        """
        self.state.estimated_reward_rep = followed_rep_estimate


    def _rate_components(self) -> tuple[float, float, float, int]:
        """The three weighted candidate drivers and the follower count."""
        return (
            float(self.state.estimated_reward_pu),
            float(self.params.gamma) * float(self.state.estimated_reward_rep),
            float(self.params.kappa) * float(self.state.estimated_reward_status),
            len(self.state.followers),
        )

    def _status_override_active(self, follower_count: int) -> bool:
        return (
            self.params.actor_rate_driver_mode
                is ActorRateDriverMode.STATUS_IF_FOLLOWERS_KAPPA0
            and np.isclose(float(self.params.kappa), 0.0, atol=1e-12, rtol=0.0)
            and follower_count >= int(self.params.actor_rate_status_override_min_followers)
        )

    def actor_rate_driver(self) -> float:
        """H_i = max{Ĵ^pu, γ Ĵ^r, κ Ĵ^s}, or the experimental status override."""
        pu, rep, status, n_followers = self._rate_components()
        if self._status_override_active(n_followers):
            return float(self.state.estimated_reward_status)
        return max(pu, rep, status)

    def actor_rate_terms(self) -> dict:
        """Full breakdown for audit rows. Not called on the production path."""
        pu, rep, status, n_followers = self._rate_components()
        standard = max(pu, rep, status)
        labels = "|".join(
            name for name, val in (("pu", pu), ("rep", rep), ("status", status))
            if np.isclose(standard, val, atol=1e-12, rtol=0.0)
        )
        override = self._status_override_active(n_followers)
        driver = float(self.state.estimated_reward_status) if override else standard
        return {
            "pu_term": pu,
            "rep_term": rep,
            "status_term": status,
            "driver": driver,
            "driver_label": "status_override" if override else labels,
            "standard_driver": standard,
            "standard_driver_label": labels,
            "status_override_active": int(override),
            "follower_count": n_followers,
            "actor_rate_driver_mode": self.params.actor_rate_driver_mode.value,
            "actor_rate_status_override_min_followers":
                int(self.params.actor_rate_status_override_min_followers),
        }

    # ==================== Section 6.7: Actor Interaction Rates ====================

    def update_actor_interaction_rate(self, alpha_rate: float):
        """
        Section 6.7: Learn actor interaction rates via Eq. (13)
        
        μ_{a,i}(t) = [μ_{a,i}(t-1) + α(t) · (-θ'(M - μ_{a,i}) · u_0 + θ'(μ_{a,i}) · H_i)]^M_0
        
        where:
        - θ(μ) = 1 - exp(-μ), so θ'(μ) = exp(-μ)
        - H_i = max{Ĵ^{pu}_i, γ Ĵ^r_i, κ Ĵ^s_i}  <-- WEIGHTED MAX
        - u_0 is utility from outside interactions
        - M is total interaction budget
        - [x]^M_0 = clip(x, 0, M)
        """
        max_reward = float(self.actor_rate_driver())

        # Compute derivatives of θ
        mu_prev = self.state.actor_interaction_rate
        theta_prime_mu = np.exp(-mu_prev)
        theta_prime_M_minus_mu = np.exp(-(self.params.M - mu_prev))

        # Update per Eq. (13)
        update_delta = -theta_prime_M_minus_mu * self.params.u_0 + theta_prime_mu * max_reward
        mu_new = mu_prev + alpha_rate * update_delta

        # Clip to [0, M]
        self.state.actor_interaction_rate = np.clip(mu_new, 0.0, self.params.M)