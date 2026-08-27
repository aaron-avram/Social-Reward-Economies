"""
Reward Functionality
"""

from abc import ABC, abstractmethod

from numpy.random import Generator
import numpy as np

from model.config import RewardParams, Dimensions, RewardModelKind

class RewardModel(ABC):
    """u_i(s, x): reward for observer i when action x is taken in state s."""

    table: np.ndarray   # shape (num_agents, num_states, num_actions)

    def __init__(self, params: RewardParams, dims: Dimensions, rng: Generator):
        self.params = params
        self.dims = dims
        table = self._build_table(params, dims, rng)
        expected = (dims.num_agents, dims.num_states, dims.num_actions)
        if not isinstance(table, np.ndarray) or table.shape != expected:
            raise TypeError(
                f"{type(self).__name__}._build_table must return ndarray {expected}, "
                f"got {getattr(table, 'shape', type(table))}"
            )
        self.table = np.ascontiguousarray(table, dtype=float)

    @abstractmethod
    def _build_table(self, params: RewardParams, dims: Dimensions,
                     rng: Generator) -> np.ndarray:
        """Return u_i(s, a) with shape (num_agents, num_states, num_actions)."""

    def observer_utility(self, observer_id: int, state: int, action: int) -> float:
        """ Compute Single Observer Utility """
        return float(self.table[observer_id, state, action])

    def observer_utilities(self, state: int, action: int) -> np.ndarray:
        """u_i(s, x) for every observer i — one realized (state, action)."""
        return np.array(self.table[:, state, action], dtype=float, copy=True)

    def actual_payoff(self, agent_id: int, state: int, action: int) -> float:
        """ Get payoff for a specific agent """
        return self.observer_utility(agent_id, state, action)


def state_probabilities(num_states):
    """ Compute State Probabilities """
    return np.ones(num_states) / num_states


class SimplePreferredAction(RewardModel):
    """
        Build Reward Table with Simple Preferred Action
    """

    def _build_table(self, params: RewardParams, dims: Dimensions, rng: Generator) -> np.ndarray:
        table = np.zeros((dims.num_agents, dims.num_states, dims.num_actions))
        table[np.arange(dims.num_agents), :, np.arange(dims.num_agents) % dims.num_actions] = 1.0

        return table


class SharedBaseGaussian(RewardModel):
    """
        Build reward table r_i(s,a) with shared base means:
        - Draw base means m(s,a) once.
        - For each agent i, draw r_i(s,a) ~ Normal(m(s,a), sigma_agent).
        - Clip to a positive range to avoid sign-related artifacts in PU baselines.
    """

    def _build_table(self, params: RewardParams, dims: Dimensions, rng: Generator) -> np.ndarray:
        base = rng.normal(params.base_mu, params.base_sigma, size=(dims.num_states, dims.num_actions))
        base = np.clip(base, params.clip_min, params.clip_max)
        t = rng.normal(base[None, :, :], params.agent_sigma, size=(dims.num_agents, dims.num_states, dims.num_actions))

        return np.clip(t, params.clip_min, params.clip_max)


class SharedGoodBadHeterogeneous(RewardModel):
    """
        Build reward table r_i(s,a) with one shared good action per state and
        agent-specific payoff heterogeneity around that shared structure.

        For each state s:
        - sample a designated good action g_hat(s);
        - draw each agent's rewards around a shared good/bad base;
        - enforce that the good action remains at least reward_order_gap above
          every bad action after sampling and clipping.
    """

    def _build_table(self, params: RewardParams, dims: Dimensions, rng: Generator) -> np.ndarray:
        clip_min = float(params.clip_min)
        clip_max = float(params.clip_max)
        gap = float(params.order_gap)
        if gap < 0.0:
            raise ValueError("reward_order_gap must be non-negative.")
        if gap >= (clip_max - clip_min):
            raise ValueError("reward_order_gap must be smaller than reward_clip_max - reward_clip_min.")

        num_states = int(dims.num_states)
        num_actions = int(dims.num_actions)
        num_agents = int(dims.num_agents)

        good_actions = rng.integers(0, num_actions, size=num_states, dtype=int)
        base = np.full((num_states, num_actions), float(params.bad_value), dtype=float)
        base[np.arange(num_states), good_actions] = float(params.good_value)

        table = rng.normal(
            loc=base[np.newaxis, :, :],
            scale=float(params.agent_sigma),
            size=(num_agents, num_states, num_actions),
        )
        table = np.clip(table, clip_min, clip_max)

        for agent_id in range(num_agents):
            for state in range(num_states):
                good_action = int(good_actions[state])
                bad_actions = [a for a in range(num_actions) if a != good_action]
                if not bad_actions:
                    continue

                good_val = float(np.clip(table[agent_id, state, good_action], clip_min + gap, clip_max))
                bad_vals = np.clip(table[agent_id, state, bad_actions], clip_min, clip_max)

                max_bad = float(np.max(bad_vals))
                if good_val < max_bad + gap:
                    good_val = min(clip_max, max_bad + gap)
                bad_cap = good_val - gap
                bad_vals = np.minimum(bad_vals, bad_cap)

                max_bad = float(np.max(bad_vals))
                if good_val < max_bad + gap:
                    good_val = min(clip_max, max_bad + gap)
                    bad_vals = np.minimum(bad_vals, good_val - gap)

                table[agent_id, state, good_action] = good_val
                table[agent_id, state, bad_actions] = bad_vals
        self._shared_good_actions = good_actions

        return table


class ConsensusWelfareGaussian(RewardModel):
    """
    Consensus Welfare Gaussian
    """

    def _build_table(self, params: RewardParams, dims: Dimensions, rng: Generator) -> np.ndarray:
        if dims.num_actions != 2:
            raise ValueError("consensus_welfare_gaussian requires num_actions=2")
        
        table = np.zeros((dims.num_agents, dims.num_states, dims.num_actions), dtype=float)

        lambda_vals = rng.uniform(
            params.lambda_min,
            params.lambda_max,
            size=dims.num_agents,
        )

        C = np.zeros((dims.num_states, dims.num_actions), dtype=float)
        W = np.zeros((dims.num_states, dims.num_actions), dtype=float)

        for s in range(dims.num_states):
            # action 0 = consensus-easy, action 1 = welfare-better
            C[s, 0] = rng.normal(params.consensus_high, 0.02)
            C[s, 1] = rng.normal(params.consensus_low, 0.02)

            W[s, 0] = rng.normal(params.welfare_low, 0.02)
            W[s, 1] = rng.normal(params.welfare_high, 0.02)

        for i in range(dims.num_agents):
            lam = lambda_vals[i]
            for s in range(dims.num_states):
                base = lam * C[s, :] + (1.0 - lam) * W[s, :]
                vals = rng.normal(
                    loc=base,
                    scale=params.agent_sigma,
                    size=dims.num_actions,
                )
                vals = np.clip(vals, params.clip_min, params.clip_max)
                table[i, s, :] = vals

        return table

REWARD_MODELS: dict[RewardModelKind, type[RewardModel]] = {
    RewardModelKind.SIMPLE_PREFERRED_ACTION:       SimplePreferredAction,
    RewardModelKind.SHARED_BASE_GAUSSIAN:          SharedBaseGaussian,
    RewardModelKind.SHARED_GOOD_BAD_HETEROGENEOUS: SharedGoodBadHeterogeneous,
    RewardModelKind.CONSENSUS_WELFARE_GAUSSIAN:    ConsensusWelfareGaussian,
}


def build_reward_model(params: RewardParams, dims: Dimensions,
                       rng: Generator) -> RewardModel:
    """
    Build reward model
    """
    return REWARD_MODELS[params.kind](params, dims, rng)