"""
Plugin interface for the unified experiment harness.

A plugin is a bundle of behaviour that can be attached to ANY experiment: it may
add CLI arguments, contribute sweep axes, act at points during a run, and add
columns to the output. Perturbations, extra instrumentation, and early-stopping
rules are all plugins.

WHY A PLUGIN AND NOT A SUBCLASS HOOK
------------------------------------
The obvious alternative is for each experiment to override a `run_protocol`
method. That works for one perturbation experiment and then stops: it cannot
express "sweep gamma AND perturbation strength in one grid", because the
override has no way to contribute an axis. Plugins can, via `axes()`, so a
perturbation sweep is an ordinary grid point rather than a separate experiment.

CONTRACT
--------
Every hook has a no-op default; implement only what you need. Hooks fire in the
order plugins appear in the experiment's `plugins` list, and that order is part
of the experiment's definition — two plugins that both mutate state on the same
step are order-dependent, so declare them deliberately.

`measure()` keys must be disjoint across plugins. The base asserts this rather
than silently letting one plugin's column overwrite another's.

STATE
-----
A plugin instance is reused across grid points and seeds. `on_run_start` must
reset everything the plugin accumulates; anything left over leaks between runs
and produces results that depend on iteration order. The base calls
`reset()` before each run as a safety net, but a plugin that keeps state in
attributes it does not clear will still be wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from model.agent import Agent, AgentRole
from model.config import SystemConfig
from model.results import SimulationResults
from model.system import MultiAgentSystem


# ============================================================================
# Sweep axes
# ============================================================================

@dataclass(frozen=True)
class Axis:
    """
    One dimension of the parameter grid.

    `name` becomes a CSV column, so it must be unique across the experiment's
    axes and every plugin's axes.

    Exactly one of `apply` or `store` is set:

      apply(cfg, value) -> SystemConfig
          The axis varies an engine parameter. Must RETURN a new config; the
          param groups are frozen, so mutation raises.

      store = "<attribute>"
          The axis varies a plugin's own parameter. The base sets
          `setattr(plugin, store, value)` before the run. Use this for things
          the engine knows nothing about, e.g. perturbation strength.
    """
    name: str
    values: Sequence[Any]
    apply: Optional[Callable[[SystemConfig, Any], SystemConfig]] = None
    store: Optional[str] = None

    def __post_init__(self):
        if (self.apply is None) == (self.store is None):
            raise ValueError(f"axis {self.name!r}: set exactly one of apply/store")
        if len(self.values) == 0:
            raise ValueError(f"axis {self.name!r}: no values to sweep")


# ============================================================================
# Run context — the entire surface a plugin may touch
# ============================================================================

@dataclass
class RunContext:
    """
    Everything a plugin can see or do, for one run.

    Deliberately the ONLY channel: a plugin that reaches for `system._private`
    is a sign the context is missing a method. (The old perturbation experiment
    read `system._shared_good_actions` directly, which broke the moment the
    reward tables moved onto the reward model.)
    """
    system: MultiAgentSystem
    args: Namespace
    point: dict[str, Any]          # this grid point, keyed by axis name
    seed: int
    t: int = 0

    # ---- read-only views -------------------------------------------------

    @property
    def results(self) -> SimulationResults:
        return self.system.results

    @property
    def agents(self) -> Sequence[Agent]:
        return self.system.agents

    def follower_counts(self) -> list[int]:
        return [len(a.state.followers) for a in self.system.agents]

    def opinion_leader(self) -> int:
        """Most-followed agent, or -1 when nobody has followers."""
        counts = self.follower_counts()
        return int(np.argmax(counts)) if max(counts, default=0) > 0 else -1

    def role_counts(self) -> dict[str, int]:
        roles = [a.state.role for a in self.system.agents]
        return {r.value: roles.count(r) for r in AgentRole}

    def norm_policy(self, agent_id: int) -> np.ndarray:
        """The (S, A) policy agent_id is currently acting on."""
        from model import welfare
        return welfare.current_policies(
            self.system.agents, self.system._leader_weights())[agent_id]

    # ---- interventions ---------------------------------------------------
    #
    # A proper method for each thing a plugin legitimately needs to change.
    # Adding one here is the right response to a plugin that wants to poke at
    # engine internals.

    def set_policy_weights(self, agent_id: int, weights: np.ndarray) -> None:
        """Overwrite an agent's behaviour weights. Shape must match (S, A)."""
        agent = self.system.agents[agent_id]
        expected = agent.state.weights_pu.shape
        if weights.shape != expected:
            raise ValueError(f"weights must be {expected}, got {weights.shape}")
        if agent.state.role is AgentRole.STATUS:
            agent.state.weights_status = np.array(weights, dtype=float, copy=True)
        else:
            agent.state.weights_pu = np.array(weights, dtype=float, copy=True)

    def push_toward_action(self, agent_id: int, action: int, strength: float) -> None:
        """
        Bias an agent's policy toward `action` by adding `strength` to that
        action's logit in every state. Replaces the old experiment's direct
        manipulation of reward tables.
        """
        agent = self.system.agents[agent_id]
        weights = agent.get_behavior_weights().copy()
        weights[:, action] += float(strength)
        self.set_policy_weights(agent_id, weights)

    def good_actions(self) -> Optional[np.ndarray]:
        """
        The designated good action per state, for reward models that define one
        (shared_good_bad_heterogeneous). None otherwise — check before using.
        """
        return getattr(self.system.rewards, "_shared_good_actions", None)

    def force_role(self, agent_id: int, role: AgentRole) -> None:
        """Set a role directly. The next role update may override it."""
        self.system.agents[agent_id].state.role = role


# ============================================================================
# Plugin
# ============================================================================

class Plugin(ABC):
    """
    Base for all plugins. Every hook is a no-op by default.

    Subclasses set `name`, which prefixes nothing automatically — plugin
    `measure()` keys are used verbatim as CSV columns, so name them
    distinctly (e.g. "perturb_recovery_step", not "recovery_step").
    """

    name: str = "plugin"

    # ---- setup -----------------------------------------------------------

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add this plugin's CLI arguments. The base calls this once."""

    def axes(self, args: Namespace) -> list[Axis]:
        """
        Sweep axes this plugin contributes. Usually `store=` axes setting the
        plugin's own parameters.

        This is the hook that makes a perturbation sweep expressible: returning
        an Axis here means the base sweeps gamma x strength as ONE grid.
        """
        return []

    def configure(self, args: Namespace, cfg: SystemConfig) -> SystemConfig:
        """
        Last chance to adjust the config before a run. Called after the
        experiment's config and after all axis `apply` functions.

        Must RETURN the config — the param groups are frozen.
        """
        return cfg

    def reset(self) -> None:
        """
        Clear per-run state. The base calls this before every run, so a plugin
        instance is safe to reuse across grid points and seeds.

        Override whenever the plugin accumulates anything.
        """

    # ---- run lifecycle ---------------------------------------------------

    def on_run_start(self, ctx: RunContext) -> None:
        """Before the first step. Record baselines here."""

    def on_step(self, ctx: RunContext) -> None:
        """After each step. `ctx.t` is the step just completed (1-based)."""

    def on_run_end(self, ctx: RunContext) -> None:
        """After the last step, before measure()."""

    def should_stop(self, ctx: RunContext) -> bool:
        """
        Return True to end the run early. The base ORs this across plugins, so
        any one plugin can stop the run — used for convergence-based stopping.
        """
        return False

    # ---- output ----------------------------------------------------------

    def measure(self, ctx: RunContext) -> dict[str, Any]:
        """
        Columns this plugin adds to the per-run CSV.

        Keys must be disjoint from the experiment's and every other plugin's.
        Return the same keys on every run, including when the plugin did
        nothing — a column that appears only sometimes produces a ragged CSV.
        """
        return {}
