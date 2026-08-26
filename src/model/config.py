"""
Configuration Classes for MultiAgent System
"""

from dataclasses import dataclass, field, asdict
from typing import List, NamedTuple
from enum import Enum


SCHEMA_VERSION: int = 1

class RewardModelKind(Enum):
    """
    Simulation Reward Model
    """
    SIMPLE_PREFERRED_ACTION = "simple_preferred_action"
    CONSENSUS_WELFARE_GAUSSIAN = "consensus_welfare_gaussian"
    SHARED_BASE_GAUSSIAN = "shared_base_gaussian"
    SHARED_GOOD_BAD_HETEROGENEOUS = "shared_good_bad_heterogeneous"

class Eq9Mode(Enum):
    """
    Eq 9 Mode
    """
    PARTICIPANTS_ONLY = "participants_only"
    ALL_AGENTS = "all_agents"

class LeaderUpdateMode(Enum):
    """
    Leader Update Mode
    """

    PARTICIPANTS_ONLY_POST_EQ9 = "participants_only_post_eq9"
    ALL_AGENTS_POST_EQ9 = "all_agents_post_eq9"
    PARTICIPANTS_ONLY_PRE_EQ9 = "participants_only_pre_eq9"

class ActorRateDriverMode(Enum):
    """
    Actor Rate Driver Mode
    """
    STANDARD = "standard"
    STATUS_IF_FOLLOWERS_KAPPA0 = "status_if_followers_kappa0"

class TrackingMode(Enum):
    """
    Tracking Mode
    """
    FULL = "full"
    LIGHT = "light"


def _unwrap(x):
    if isinstance(x, Enum):  return x.value
    if isinstance(x, dict):  return {k: _unwrap(v) for k, v in x.items()}
    if isinstance(x, list):  return [_unwrap(v) for v in x]
    return x

@dataclass(frozen=True)
class Dimensions:
    """
    Dimensions
    """
    num_agents: int = 6
    num_states: int = 3
    num_actions: int = 2


@dataclass(frozen=True)
class Stepsize:
    """
    Step size tracker
    """
    base: float
    decay: float

    def at(self, t: int) -> float:
        """
        Track value at time step t
        """
        return self.base / (1.0 + t * self.decay)


@dataclass(frozen=True)
class AlgorithmParams:
    """
    Algorithm Setup Parameters
    """
    gamma: float = 2.0 # Reputation Weight
    kappa: float = 2.0 # Status Weight

    c_threshold: float = 0.1 # minimum fraction of followers needed for status

    B_R: float = 0.8  # Reputation threshold to START following
    B_F: float = 0.6  # Reputation threshold to CONTINUE following (B_F < B_R)

    delta: float = 0.1  # Tolerance for near-ties in reputation

    M: float = 1.0  # Total interaction rate budget
    u_0: float = 0.1  # Utility from outside interactions

    gossip_rate: float = 0.5  # Probability of gossip at each step
    gossip_alpha: float = 0.5  # Averaging parameter in gossip

    initial_actor_interaction_rate: float = 0.7
    initial_participant_interaction_rate: float = 0.7
    actor_rate_status_override_min_followers: int = 10

    # modes
    actor_rate_driver_mode: ActorRateDriverMode = ActorRateDriverMode.STANDARD
    eq9_averaging_mode: Eq9Mode = Eq9Mode.PARTICIPANTS_ONLY
    leader_update_mode: LeaderUpdateMode = LeaderUpdateMode.PARTICIPANTS_ONLY_POST_EQ9

    def __post_init__(self):
        if not (0.0 <= self.B_F < self.B_R <= 1.0):
            raise ValueError(f"need 0 <= B_F < B_R <= 1, got {self.B_F=}, {self.B_R=}")
        if not (0.0 <= self.c_threshold <= 1.0):
            raise ValueError(f"c_threshold must be in [0,1], got {self.c_threshold}")
        if self.gamma < 0 or self.kappa < 0:
            raise ValueError("gamma and kappa must be non-negative")

@dataclass(frozen=True)
class RewardParams:
    """
    Reward Simulation Parameters
    """

    kind: RewardModelKind = RewardModelKind.SIMPLE_PREFERRED_ACTION
    base_mu: float = 0.5
    base_sigma: float = 0.08
    agent_sigma: float = 0.03
    clip_min: float = 0.01
    clip_max: float = 2.5
    good_value: float = 1.0
    bad_value: float = 0.1
    order_gap: float = 0.02

    consensus_high: float = 0.95
    consensus_low: float = 0.45
    welfare_high: float = 1.05
    welfare_low: float = 0.35

    lambda_min: float = 0.55
    lambda_max: float = 0.85

    def __post_init__(self):
        if self.order_gap < 0.0:
            raise ValueError("reward_order_gap must be non-negative.")
        if self.order_gap >= (self.clip_max - self.clip_min):
            raise ValueError("reward_order_gap must be smaller than reward_clip_max - reward_clip_min.")

class ResolvedStepsizes(NamedTuple):
    """
    at t values
    """
    alpha_pu: float
    beta_status: float
    eta_v: float
    eta_s: float
    eta_J: float
    alpha_rate: float

@dataclass(frozen=True)
class StepsizeParams:
    """
    Simulation Step size parameters
    """
    alpha_pu: Stepsize = field(default_factory=lambda: Stepsize(0.05, 0.01))
    beta_status: Stepsize = field(default_factory=lambda: Stepsize(0.10, 0.01))
    eta_v: Stepsize = field(default_factory=lambda: Stepsize(0.10, 0.01))
    eta_s: Stepsize = field(default_factory=lambda: Stepsize(0.10, 0.01))
    eta_J: Stepsize = field(default_factory=lambda: Stepsize(0.05, 0.01))
    alpha_rate: Stepsize = field(default_factory=lambda: Stepsize(0.01, 0.005))

    def at(self, t: int) -> ResolvedStepsizes:
        """
        Tracks value at time t
        """
        t = max(1, int(t))
        return ResolvedStepsizes(
            alpha_pu=self.alpha_pu.at(t),
            beta_status=self.beta_status.at(t),
            eta_v=self.eta_v.at(t),
            eta_s=self.eta_s.at(t),
            eta_J=self.eta_J.at(t),
            alpha_rate=self.alpha_rate.at(t),
        )

@dataclass(frozen=True)
class RuntimeParams:
    """
    Simulation Runtime parameters
    """
    seed: int = 0
    tracking_mode: TrackingMode = TrackingMode.FULL
    use_numpy_fast_path: bool = False  # Enable vectorized reputation updates for large-N sweeps
    force_all_active_debug: bool = False  # Debug override: force A_a(t)=A_p(t)=C every step
    num_time_steps: int = 2000

@dataclass(frozen=True)
class ScheduleParams:
    """
    Scheduling Parameters
    """
    role_update_s0: int = 0
    role_update_T_sequence: List[int] = field(default_factory=list)
    role_update_base_interval: int = 50  # Base interval for constant/increasing schedules
    fixed_role_update_interval: bool = False  # If True, use constant spacing T_n = const
    role_update_epochs: List[int] = field(default_factory=list)  # Optional direct s_n list (alternative input)

@dataclass
class SystemConfig:
    """Section 6–7 configuration with all required parameters"""

    dims: Dimensions = field(default_factory=Dimensions)
    algorithm: AlgorithmParams = field(default_factory=AlgorithmParams)
    reward: RewardParams = field(default_factory=RewardParams)
    stepsizes: StepsizeParams = field(default_factory=StepsizeParams)
    runtime: RuntimeParams = field(default_factory=RuntimeParams)
    schedule: ScheduleParams = field(default_factory=ScheduleParams)

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **_unwrap(asdict(self))}