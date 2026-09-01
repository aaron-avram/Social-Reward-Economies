"""
Experiment D: perturbation and recovery.

This is the design's stress test. Unlike A, B and C -- which are pure sweeps
whose measurements are functions of the finished history -- D is a state machine
that intervenes *during* the run:

    observe -> detect convergence -> perturb the leader -> watch for recovery

It needs `before_step` (interventions must land before the engine steps),
`on_step` (streak counters and per-step diagnostics), and per-run time series
surfaced at sweep time for the trajectory figures. If the plugin protocol can
express D, it can express anything the other three need.

The perturbation operators below are ported verbatim from
src/experiments/v2/perturbation_recovery.py. They are pure functions of
(system, leader_id, strength), which is what lets them be unit tested and
swapped without touching the state machine.

WHAT THE PERTURBATION ACTUALLY IS
---------------------------------
Only the leader's POLICY WEIGHTS are overwritten, and only for the duration of
the perturbation window -- the shock is re-applied every perturbed step and
released automatically when the window closes. Nothing forces followers to
leave. They leave only if the resulting observed utilities push their Section-7
follow decision below the personal-utility alternative. That is what makes the
recovery measurement meaningful: both the collapse and the recovery are
endogenous.

`--collapse-followers-on-perturb` breaks this property deliberately: it
dissolves the follower set directly, which measures re-formation from scratch
rather than resilience to a bad leader. The two are different questions; do not
read one as the other.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.agent import AgentRole

from experiments.harness.cli import parse_csv_ints
from experiments.harness.plugins import RunContext, RunPlugin, SweepContext, SweepPlugin


# ------------------------------------------------------- threshold helpers ---

def resolve_threshold(value: Optional[float], n_minus_1: int, default_abs: float) -> int:
    """A threshold given as <= 1.0 is a ratio of (N-1); above 1.0 it is absolute."""
    if value is None:
        return int(np.ceil(default_abs))
    if value <= 1.0:
        return int(np.ceil(value * n_minus_1))
    return int(np.ceil(value))


def derive_interval_scaled_windows(role_update_interval: int) -> Dict[str, int]:
    """Keep the timing windows proportional to the role-update cadence.

    Convergence, perturbation and recovery are all measured in steps, but the
    only thing that can change an agent's role is a role update. Windows fixed
    in steps therefore mean something different at every cadence; these ratios
    hold them fixed in role-update epochs instead.
    """
    interval = max(1, int(role_update_interval))
    return {
        "perturb_duration": int(3 * interval),
        "conv_hold_steps": int(np.ceil(1.2 * interval)),
        "recovery_hold_steps": int(np.ceil(0.8 * interval)),
        "stable_tail_window": int(2 * interval),
    }


def longest_true_run(mask: Sequence[bool]) -> int:
    best = cur = 0
    for m in mask:
        if bool(m):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def compute_normless_duration(
    top_series: Sequence[float],
    dominant_threshold: float,
    *,
    start_idx: int,
    end_idx: int,
) -> int:
    """Longest stretch with no dominant leader, within [start_idx, end_idx]."""
    if len(top_series) == 0:
        return 0
    start_idx = max(0, int(start_idx))
    end_idx = min(len(top_series) - 1, int(end_idx))
    if end_idx < start_idx:
        return 0
    arr = np.asarray(top_series, dtype=float)[start_idx:end_idx + 1]
    return longest_true_run(arr < float(dominant_threshold))


def compute_alt_leader_stats(
    follower_counts: Sequence[int], ex_leader_id: int
) -> Tuple[int, int]:
    """The strongest follower bloc that is not the ex-leader's, and whose it is."""
    best_id, best_followers = -1, 0
    for agent_id, count in enumerate(follower_counts):
        if int(agent_id) == int(ex_leader_id):
            continue
        if int(count) > best_followers:
            best_followers = int(count)
            best_id = int(agent_id)
    return best_id, best_followers


def summarize_positive_step1_margins(
    margins: Sequence[float],
) -> Tuple[float, float]:
    arr = np.asarray(margins, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    positive = arr > 0.0
    share = float(np.mean(positive))
    mean_positive = float(np.mean(arr[positive])) if np.any(positive) else float("nan")
    return share, mean_positive


def detect_first_threshold_timestep(
    series: Sequence[float], threshold: float, *, start_idx: int = 0
) -> int:
    start_idx = max(0, int(start_idx))
    for idx in range(start_idx, len(series)):
        if float(series[idx]) >= float(threshold):
            return int(idx + 1)
    return -1


# ------------------------------------------------- perturbation operators ---

def apply_low_payoff_perturbation(agent, strength: float) -> None:
    """Force the leader's policy to strongly prefer a non-preferred action."""
    strength = float(abs(strength))
    num_states, num_actions = (int(x) for x in agent.state.weights_pu.shape)

    forced = np.full((num_states, num_actions), -strength, dtype=float)
    pref = int(agent.preferred_action % max(1, num_actions))
    if num_actions >= 2:
        anti = int((pref + 1) % num_actions)
        forced[:, anti] = strength
        forced[:, pref] = -strength

    agent.state.weights_pu = forced.copy()
    agent.state.weights_status = forced.copy()


def apply_targeted_low_payoff_perturbation(
    system, leader_id: int, *, strength: float, target_ids: Sequence[int]
) -> None:
    """In each state, force the leader toward the action that is worst for the targets.

    Stronger than the untargeted version but still endogenous: only the leader's
    policy changes.
    """
    agent = system.agents[leader_id]
    target_ids = [
        int(i) for i in target_ids
        if 0 <= int(i) < len(system.agents) and int(i) != leader_id
    ]
    if not target_ids:
        apply_low_payoff_perturbation(agent, strength)
        return

    strength = float(abs(strength))
    num_states, num_actions = (int(x) for x in agent.state.weights_pu.shape)
    forced = np.full((num_states, num_actions), -strength, dtype=float)

    for state in range(num_states):
        scores = [
            float(np.mean([
                system.rewards.observer_utility(observer_id, state, action)
                for observer_id in target_ids
            ]))
            for action in range(num_actions)
        ]
        forced[state, int(np.argmin(scores))] = strength

    agent.state.weights_pu = forced.copy()
    agent.state.weights_status = forced.copy()


def apply_force_bad_action_perturbation(system, leader_id: int, *, strength: float) -> None:
    """Force the leader away from the shared good action g_hat(s) in every state.

    Only defined for the shared_good_bad_heterogeneous reward model, which is the
    only one carrying a system-wide good action per state.
    """
    good_actions = getattr(system.rewards, "_shared_good_actions", None)
    if good_actions is None:
        raise ValueError(
            "force_bad_action requires the shared_good_bad_heterogeneous reward model."
        )

    agent = system.agents[leader_id]
    strength = float(abs(strength))
    num_states, num_actions = (int(x) for x in agent.state.weights_pu.shape)

    forced = np.full((num_states, num_actions), strength, dtype=float)
    for state in range(num_states):
        forced[state, int(good_actions[state])] = -strength

    agent.state.weights_pu = forced.copy()
    agent.state.weights_status = forced.copy()


def apply_reputation_shock(system, leader_id: int, factor: float) -> None:
    """Multiplicative shock on every agent's reputation estimate of the leader."""
    factor = float(factor)
    if leader_id < 0 or factor >= 1.0:
        return

    for agent in system.agents:
        cur = float(agent.state.reputation_estimates.get(leader_id, 0.0))
        agent.state.reputation_estimates[leader_id] = factor * cur

    # Keep the dense matrix consistent with the per-agent dicts, or the fast
    # path and the slow path will silently disagree from here on.
    s_matrix = getattr(system.rep, "s", None)
    if s_matrix is not None:
        s_matrix[:, leader_id] *= factor


def collapse_leader_followership(system, leader_id: int) -> None:
    """Dissolve the leader's follower set outright. See the module docstring."""
    if leader_id < 0 or leader_id >= len(system.agents):
        return

    leader = system.agents[leader_id]
    for follower_id in list(leader.state.followers):
        follower = system.agents[follower_id]
        follower.state.following = None
        follower.state.role = AgentRole.PERSONAL_UTILITY
        follower.state.was_following = False
    leader.state.followers.clear()


def compute_step1_diagnostic_terms(system, agent_id: int) -> Dict[str, Any]:
    """Mirror the role-update Step-1 comparison for one agent.

    Diagnostics must use the same currently selected non-self target as the
    role-update logic, not a self-inclusive maximum over reputation estimates --
    otherwise the reported margin is not the margin the agent actually acted on.
    """
    agent = system.agents[agent_id]
    algo = system.config.algorithm

    in_C = len(agent.state.followers) == 0
    in_R = agent.state.role == AgentRole.REPUTATION
    hysteresis_active = bool(in_C and in_R and float(algo.B_F) < float(algo.B_R))
    B_i = float(algo.B_F) if hysteresis_active else float(algo.B_R)

    target_k = agent.state.highest_rep_agent_estimate
    selected_rep_raw = (
        float(agent.state.reputation_estimates.get(target_k, 0.0))
        if target_k is not None else 0.0
    )

    selected_rep_weighted = float(algo.gamma) * selected_rep_raw
    est_pu = float(agent.state.estimated_reward_pu)
    threshold = max(B_i, est_pu)

    return {
        "target_id": None if target_k is None else int(target_k),
        "hysteresis_active": hysteresis_active,
        "selected_rep_raw": selected_rep_raw,
        "selected_rep_weighted": selected_rep_weighted,
        "estimated_reward_pu": est_pu,
        "effective_threshold": B_i,
        "threshold": threshold,
        "step1_margin": float(selected_rep_weighted - threshold),
    }


# ------------------------------------------------------ the state machine ---

RECORD_COLUMNS = (
    "converged", "t_conv", "leader_pre", "pre_followers",
    "t_perturb_start", "t_perturb_end",
    "drop_min", "drop_fraction", "time_to_drop",
    "normless_duration", "pu_share_peak_during_drop",
    "recovery_time", "leader_post_recovery", "leader_changed",
    "stable_recovery", "stable_tail_window",
    "welfare_pre", "welfare_drop", "welfare_recovered",
    "final_leader", "final_leader_changed", "final_top_followers",
    "post_perturb_role_updates_available", "max_alt_leader_followers_post",
    "time_to_alt_leader_25pct", "time_to_alt_leader_50pct",
    "time_to_alt_leader_75pct",
    "final_share_positive_step1_margin", "final_pu_share",
)


class PerturbationRecoveryPlugin(RunPlugin):
    """Converge, perturb the leader, watch whether leadership recovers."""

    name = "perturbation_recovery"
    columns = RECORD_COLUMNS

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("perturbation")
        g.add_argument("--perturb-strength", type=float, default=8.0)
        g.add_argument("--perturb-duration", type=int, default=600)
        g.add_argument(
            "--perturb-policy-mode",
            choices=["targeted_low_payoff", "force_bad_action"],
            default="targeted_low_payoff",
        )
        g.add_argument(
            "--collapse-followers-on-perturb",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Dissolve the pre-leader's follower set at perturbation start. "
                 "This makes the measurement re-formation-from-scratch rather "
                 "than resilience; see the module docstring.",
        )
        g.add_argument(
            "--reputation-shock-factor",
            type=float,
            default=1.0,
            help="Per-perturbed-step multiplicative shock on s_i(leader, t). "
                 "1.0 disables.",
        )
        g.add_argument("--post-window", type=int, default=2500)

        g = parser.add_argument_group("convergence / recovery criteria")
        g.add_argument(
            "--conv-threshold", type=float, default=None,
            help="Top-follower count defining convergence. <= 1 is a ratio of (N-1).",
        )
        g.add_argument("--conv-hold-steps", type=int, default=200)
        g.add_argument(
            "--recovery-threshold", type=float, default=0.9,
            help="Top-follower count defining recovery. <= 1 is a ratio of (N-1).",
        )
        g.add_argument("--recovery-hold-steps", type=int, default=150)
        g.add_argument(
            "--stable-tail-window", type=int, default=200,
            help="Require strict convergence over the last K steps to count as "
                 "a stable recovery.",
        )
        g.add_argument(
            "--dominant-threshold", type=float, default=0.5,
            help="Dominance threshold for normlessness. <= 1 is a ratio of (N-1).",
        )
        g.add_argument(
            "--drop-fraction-threshold", type=float, default=0.5,
            help="Time-to-drop threshold, as a fraction of pre-perturbation followers.",
        )

    # -- setup -------------------------------------------------------------

    def on_start(self, system, ctx: RunContext) -> None:
        args = ctx.args
        n = int(args.num_agents)
        n_minus_1 = max(1, n - 1)

        ctx.scratch["pr"] = st = {
            "n": n,
            "n_minus_1": n_minus_1,
            "conv_threshold": resolve_threshold(
                args.conv_threshold, n_minus_1, default_abs=n_minus_1),
            "recovery_threshold": resolve_threshold(
                args.recovery_threshold, n_minus_1, default_abs=0.9 * n_minus_1),
            "dominant_threshold": resolve_threshold(
                args.dominant_threshold, n_minus_1, default_abs=0.5 * n_minus_1),
            "conv_hold": max(1, int(args.conv_hold_steps)),
            "recovery_hold": max(1, int(args.recovery_hold_steps)),
            "stable_tail_window": max(1, int(args.stable_tail_window)),
            "perturb_duration": max(1, int(args.perturb_duration)),
            "post_window": max(1, int(args.post_window)),

            "t_conv": -1,
            "leader_pre": -1,
            "pre_followers": 0,
            "t_perturb_start": -1,
            "t_perturb_end": -1,
            "leader_post_recovery": -1,
            "recovery_time": -1,
            "perturb_target_ids": [],
            "conv_streak": 0,
            "recovery_streak": 0,
            "collapse_applied": False,
            "role_updates_since_perturb_end": 0,

            "top": [], "leader": [], "alt_leader": [], "alt_followers": [],
            "ex_leader_followers": [], "pu_share": [], "welfare": [],
            "share_positive_margin": [], "mean_positive_margin": [],
            "follower_rows": [],
        }
        st["exit_diagnostics"] = []

    # -- intervention ------------------------------------------------------

    def before_step(self, system, ctx: RunContext, next_step: int) -> None:
        st = ctx.scratch["pr"]
        args = ctx.args

        in_window = (
            st["leader_pre"] >= 0
            and st["t_perturb_start"] > 0
            and st["t_perturb_start"] <= next_step <= st["t_perturb_end"]
        )
        if not in_window:
            return

        leader_pre = st["leader_pre"]
        if not st["perturb_target_ids"]:
            targets = sorted(int(i) for i in system.agents[leader_pre].state.followers)
            st["perturb_target_ids"] = targets or [
                i for i in range(st["n"]) if i != leader_pre
            ]

        if (bool(args.collapse_followers_on_perturb)
                and not st["collapse_applied"]
                and next_step == st["t_perturb_start"]):
            collapse_leader_followership(system, leader_pre)
            st["collapse_applied"] = True

        if float(args.reputation_shock_factor) < 1.0:
            apply_reputation_shock(system, leader_pre, float(args.reputation_shock_factor))

        if str(args.perturb_policy_mode) == "force_bad_action":
            apply_force_bad_action_perturbation(
                system, leader_pre, strength=float(args.perturb_strength))
        else:
            apply_targeted_low_payoff_perturbation(
                system, leader_pre,
                strength=float(args.perturb_strength),
                target_ids=st["perturb_target_ids"],
            )

    # -- observation -------------------------------------------------------

    def on_step(self, system, ctx: RunContext, step: int, role_updated: bool) -> None:
        st = ctx.scratch["pr"]
        args = ctx.args
        n = st["n"]

        followers = [len(a.state.followers) for a in system.agents]
        st["follower_rows"].append(followers)

        top_followers = int(max(followers)) if followers else 0
        leader = int(np.argmax(followers)) if top_followers > 0 else -1
        pu_share = float(
            sum(1 for a in system.agents if a.state.role == AgentRole.PERSONAL_UTILITY) / n
        )
        welfare = float(system.results.social_welfare[-1])
        alt_leader_id, alt_leader_followers = compute_alt_leader_stats(
            followers, st["leader_pre"] if st["leader_pre"] >= 0 else -1
        )

        st["top"].append(float(top_followers))
        st["leader"].append(leader)
        st["alt_leader"].append(int(alt_leader_id))
        st["alt_followers"].append(float(alt_leader_followers))
        st["pu_share"].append(pu_share)
        st["welfare"].append(welfare)

        # Phase 1: detect convergence and schedule the perturbation window.
        if st["t_conv"] < 0:
            st["conv_streak"] = (
                st["conv_streak"] + 1 if top_followers >= st["conv_threshold"] else 0
            )
            if st["conv_streak"] >= st["conv_hold"]:
                st["t_conv"] = int(system.time_step)
                st["leader_pre"] = leader
                st["pre_followers"] = top_followers
                self._set_preleader(system, leader)
                if st["t_conv"] < int(getattr(args, "num_steps_max")):
                    st["t_perturb_start"] = st["t_conv"] + 1
                    st["t_perturb_end"] = min(
                        int(args.num_steps_max),
                        st["t_perturb_start"] + st["perturb_duration"] - 1,
                    )

        st["ex_leader_followers"].append(
            float(followers[st["leader_pre"]]) if st["leader_pre"] >= 0 else float("nan")
        )

        # Phase 3: watch for recovery once the window has closed.
        if st["t_perturb_end"] > 0 and int(system.time_step) >= st["t_perturb_end"] + 1:
            if role_updated:
                st["role_updates_since_perturb_end"] += 1
            st["recovery_streak"] = (
                st["recovery_streak"] + 1
                if top_followers >= st["recovery_threshold"] else 0
            )
            if st["recovery_time"] < 0 and st["recovery_streak"] >= st["recovery_hold"]:
                st["recovery_time"] = int(system.time_step)
                st["leader_post_recovery"] = leader

        self._record_exit_diagnostics(system, ctx, st, leader, top_followers,
                                      alt_leader_id, alt_leader_followers,
                                      pu_share, followers, role_updated)

    def _set_preleader(self, system, leader_id: int) -> None:
        rec = getattr(system, "rec", None)
        if rec is not None and hasattr(rec, "preleader_id"):
            rec.preleader_id = None if leader_id < 0 else int(leader_id)

    def _record_exit_diagnostics(self, system, ctx, st, leader, top_followers,
                                 alt_leader_id, alt_leader_followers, pu_share,
                                 followers, role_updated) -> None:
        """Per-step Step-1 margin diagnostics across the perturbation window.

        The point is to distinguish "followers left because the leader became
        bad" from "followers left because personal utility became better", which
        the aggregate follower count alone cannot separate.
        """
        args = ctx.args
        leader_pre = st["leader_pre"]
        if leader_pre < 0 or st["t_perturb_start"] <= 0:
            return

        diag_end = min(int(args.num_steps_max),
                       int(st["t_perturb_end"]) + int(st["post_window"]))
        if not (st["t_perturb_start"] <= int(system.time_step) <= diag_end):
            return

        tracked_ids = st["perturb_target_ids"] or [
            i for i in range(st["n"]) if i != leader_pre
        ]

        margins, thresholds, rep_to_leader, gamma_max_reps = [], [], [], []
        highest_pre = follow_pre = pu_count = rep_other_count = target_count = 0

        for agent_id in tracked_ids:
            if agent_id == leader_pre:
                continue
            agent = system.agents[agent_id]
            target_count += 1

            terms = compute_step1_diagnostic_terms(system, agent_id)
            thresholds.append(float(terms["threshold"]))
            margins.append(float(terms["step1_margin"]))
            rep_to_leader.append(
                float(agent.state.reputation_estimates.get(leader_pre, 0.0)))
            gamma_max_reps.append(float(terms["selected_rep_weighted"]))
            highest_pre += int(terms["target_id"] == leader_pre)
            follow_pre += int(agent.state.following == leader_pre)
            pu_count += int(agent.state.role == AgentRole.PERSONAL_UTILITY)
            rep_other_count += int(
                agent.state.role == AgentRole.REPUTATION
                and agent.state.following is not None
                and agent.state.following != leader_pre
            )

        share_pos, mean_pos = summarize_positive_step1_margins(margins)
        st["share_positive_margin"].append(share_pos)
        st["mean_positive_margin"].append(mean_pos)

        def _mean(xs):
            return float(np.mean(xs)) if xs else float("nan")

        st["exit_diagnostics"].append({
            "step": int(system.time_step),
            "role_update_step": int(role_updated),
            "leader_pre": int(leader_pre),
            "leader_pre_followers": int(followers[leader_pre]),
            "top_followers": int(top_followers),
            "current_leader": int(leader),
            "alt_leader_id": int(alt_leader_id),
            "largest_alt_leader_followers": int(alt_leader_followers),
            "pu_share": float(pu_share),
            "tracked_targets": int(target_count),
            "targets_following_preleader": int(follow_pre),
            "targets_in_pu": int(pu_count),
            "targets_in_rep_elsewhere": int(rep_other_count),
            "mean_gamma_selected_rep": _mean(gamma_max_reps),
            "mean_rep_to_preleader": _mean(rep_to_leader),
            "mean_estimated_reward_pu": _mean([
                system.agents[i].state.estimated_reward_pu
                for i in tracked_ids if i != leader_pre
            ]) if target_count > 0 else float("nan"),
            "mean_threshold": _mean(thresholds),
            "mean_step1_margin": _mean(margins),
            "min_step1_margin": float(np.min(margins)) if margins else float("nan"),
            "share_positive_step1_margin": float(share_pos),
            "mean_positive_step1_margin": float(mean_pos),
            "mean_highest_rep_is_preleader": (
                float(highest_pre / target_count) if target_count > 0 else float("nan")
            ),
            "num_role_updates_since_perturb_end": int(
                st["role_updates_since_perturb_end"]),
        })

    # -- measurement -------------------------------------------------------

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        st = ctx.scratch["pr"]
        args = ctx.args
        n_minus_1 = st["n_minus_1"]

        top_arr = np.array(st["top"], dtype=float)
        alt_follower_arr = np.array(st["alt_followers"], dtype=float)
        ex_arr = np.array(st["ex_leader_followers"], dtype=float)
        pu_arr = np.array(st["pu_share"], dtype=float)
        welfare_arr = np.array(st["welfare"], dtype=float)

        final_followers = st["follower_rows"][-1] if st["follower_rows"] else [0] * st["n"]
        final_top_followers = int(max(final_followers)) if final_followers else 0
        final_leader = int(np.argmax(final_followers)) if final_top_followers > 0 else -1

        stable_tail = (
            top_arr[-min(st["stable_tail_window"], len(top_arr)):]
            if len(top_arr) > 0 else np.array([], dtype=float)
        )
        stable_recovery = bool(
            stable_tail.size > 0 and np.all(stable_tail >= st["conv_threshold"])
        )

        out = {
            "drop_min": float("nan"),
            "drop_fraction": float("nan"),
            "time_to_drop": -1,
            "normless_duration": 0,
            "pu_share_peak_during_drop": float("nan"),
            "welfare_pre": float("nan"),
            "welfare_drop": float("nan"),
            "welfare_recovered": float("nan"),
            "post_perturb_role_updates_available": 0,
            "max_alt_leader_followers_post": 0,
            "time_to_alt_leader_25pct": -1,
            "time_to_alt_leader_50pct": -1,
            "time_to_alt_leader_75pct": -1,
            "final_share_positive_step1_margin": float("nan"),
            "final_pu_share": float(pu_arr[-1]) if pu_arr.size > 0 else float("nan"),
        }

        if st["t_perturb_start"] > 0 and st["leader_pre"] >= 0:
            start_idx = st["t_perturb_start"] - 1
            end_idx = len(top_arr) - 1

            ex_window = ex_arr[start_idx:end_idx + 1]
            valid_ex = ex_window[np.isfinite(ex_window)]
            if valid_ex.size > 0:
                out["drop_min"] = float(np.min(valid_ex))
                if st["pre_followers"] > 0:
                    out["drop_fraction"] = float(
                        (st["pre_followers"] - out["drop_min"]) / st["pre_followers"])
                    target = float(st["pre_followers"]) * float(args.drop_fraction_threshold)
                    below = np.where(valid_ex <= target)[0]
                    if below.size > 0:
                        out["time_to_drop"] = int(st["t_perturb_start"] + int(below[0]))

            out["normless_duration"] = compute_normless_duration(
                top_arr, st["dominant_threshold"],
                start_idx=start_idx, end_idx=end_idx,
            )
            out["pu_share_peak_during_drop"] = float(np.max(pu_arr[start_idx:end_idx + 1]))

            tail = max(1, int(args.tail_window))
            pre_end = max(0, start_idx - 1)
            pre_start = max(0, pre_end - tail + 1)
            if pre_end >= pre_start:
                out["welfare_pre"] = float(np.mean(welfare_arr[pre_start:pre_end + 1]))

            drop_end = (
                max(start_idx, min(len(welfare_arr) - 1, st["t_perturb_end"] - 1))
                if st["t_perturb_end"] > 0 else end_idx
            )
            out["welfare_drop"] = float(np.mean(welfare_arr[start_idx:drop_end + 1]))

            post_start_idx = min(len(alt_follower_arr), max(0, int(st["t_perturb_end"])))
            post_alt_window = alt_follower_arr[post_start_idx:end_idx + 1]
            if post_alt_window.size > 0:
                out["max_alt_leader_followers_post"] = int(np.max(post_alt_window))
                for pct, key in ((0.25, "time_to_alt_leader_25pct"),
                                 (0.50, "time_to_alt_leader_50pct"),
                                 (0.75, "time_to_alt_leader_75pct")):
                    out[key] = detect_first_threshold_timestep(
                        alt_follower_arr,
                        threshold=float(np.ceil(pct * n_minus_1)),
                        start_idx=post_start_idx,
                    )

            out["post_perturb_role_updates_available"] = int(
                st["role_updates_since_perturb_end"])
            if st["share_positive_margin"]:
                out["final_share_positive_step1_margin"] = float(
                    st["share_positive_margin"][-1])

        if st["recovery_time"] > 0:
            rec_idx = st["recovery_time"] - 1
            rec_end = min(len(welfare_arr) - 1,
                          rec_idx + max(1, int(args.tail_window)) - 1)
            out["welfare_recovered"] = float(np.mean(welfare_arr[rec_idx:rec_end + 1]))

        leader_changed = bool(
            st["recovery_time"] > 0 and st["leader_post_recovery"] >= 0
            and st["leader_pre"] >= 0
            and st["leader_post_recovery"] != st["leader_pre"]
        )
        final_leader_changed = bool(
            st["leader_pre"] >= 0 and final_leader >= 0
            and final_leader != st["leader_pre"]
        )

        # Diagnostics go out as a side table keyed by seed so the sweep plugin
        # can write one file per seed, matching the v2 layout.
        for row in st["exit_diagnostics"]:
            ctx.emit("exit_diagnostics", {"seed": int(ctx.seed), **row})

        out.update({
            "converged": bool(st["t_conv"] > 0),
            "t_conv": int(st["t_conv"]),
            "leader_pre": int(st["leader_pre"]),
            "pre_followers": int(st["pre_followers"]),
            "t_perturb_start": int(st["t_perturb_start"]),
            "t_perturb_end": int(st["t_perturb_end"]),
            "recovery_time": int(st["recovery_time"]),
            "leader_post_recovery": int(st["leader_post_recovery"]),
            "leader_changed": leader_changed,
            "stable_recovery": stable_recovery,
            "stable_tail_window": int(min(st["stable_tail_window"], len(top_arr))),
            "final_leader": int(final_leader),
            "final_leader_changed": final_leader_changed,
            "final_top_followers": int(final_top_followers),
        })
        return out


# ------------------------------------------------------------- reporting ---

def format_num_for_name(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:g}".replace(".", "p").replace("-", "m")


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return re.sub(r"_+", "_", s).strip("_")


def build_run_subdir_name(args) -> str:
    """A directory name that records the parameters that produced the run.

    Two runs with different perturbation settings otherwise overwrite each
    other's CSVs, which is a quiet way to lose a result.
    """
    if str(args.run_label).strip():
        return slugify(args.run_label)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    explicit = parse_csv_ints(getattr(args, "selected_seeds", "") or "")
    if explicit:
        seed_suffix = "seedset_" + "_".join(str(s) for s in explicit)
    else:
        first = int(args.seed_start)
        seed_suffix = f"seed{first}to{first + int(args.seeds) - 1}"

    return (
        f"{slugify(args.output_prefix)}"
        f"_{args.mode}"
        f"_g{format_num_for_name(args.gamma)}"
        f"_k{format_num_for_name(args.kappa)}"
        f"_N{int(args.num_agents)}"
        f"_S{int(args.num_states)}"
        f"_steps{int(args.num_steps_max)}"
        f"_{seed_suffix}"
        f"_{stamp}"
    )


class PerturbationFigures(SweepPlugin):
    """Per-seed trajectory plots and per-seed diagnostic CSVs.

    Reaches into `ctx.run_contexts` because the figures need each run's full
    time series, which is not row-shaped and does not belong in a side table.
    """

    name = "perturbation_figures"

    def figures(self, ctx: SweepContext) -> None:
        from experiments.harness.tables import write_csv

        args = ctx.args
        prefix = str(args.output_prefix)
        single_run = len(ctx.run_contexts) == 1

        by_seed: Dict[int, List[Dict[str, Any]]] = {}
        for row in ctx.side_tables.get("exit_diagnostics", []):
            by_seed.setdefault(int(row["seed"]), []).append(row)

        for run_ctx, record in zip(ctx.run_contexts, ctx.records):
            seed = int(run_ctx.seed)
            st = run_ctx.scratch["pr"]

            png = (ctx.output_dir / f"{prefix}.png" if single_run
                   else ctx.output_dir / f"{prefix}_seed{seed}_{ctx.mode}.png")
            _plot_seed_trajectory(
                output_file=png,
                sample_interval=int(args.plot_sample_interval),
                top_series=np.array(st["top"], dtype=float),
                ex_leader_series=np.array(st["ex_leader_followers"], dtype=float),
                pu_share_series=np.array(st["pu_share"], dtype=float),
                welfare_series=np.array(st["welfare"], dtype=float),
                record=record,
            )

            rows = by_seed.get(seed, [])
            if rows:
                write_csv(
                    ctx.output_dir / f"{prefix}_seed{seed}_{ctx.mode}_exit_diagnostics.csv",
                    [{k: v for k, v in r.items() if k != "seed"} for r in rows],
                )


def _downsample(x: np.ndarray, y: np.ndarray, sample_interval: int):
    step = max(1, int(sample_interval))
    if len(x) <= step:
        return x, y
    return x[::step], y[::step]


def _plot_seed_trajectory(
    *,
    output_file: Path,
    sample_interval: int,
    top_series: np.ndarray,
    ex_leader_series: np.ndarray,
    pu_share_series: np.ndarray,
    welfare_series: np.ndarray,
    record: Dict[str, Any],
) -> None:
    """Four stacked panels sharing a time axis, with the phase boundaries marked.

    The vertical guides are the point: a follower-count dip means nothing until
    you can see whether it sits inside the perturbation window or after it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if top_series.size == 0:
        return

    t = np.arange(1, len(top_series) + 1, dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.0), sharex=True)

    panels = [
        (top_series, "Top followers", "top followers"),
        (ex_leader_series, "Pre-perturbation leader's followers", "followers"),
        (pu_share_series, "Share of agents in PERSONAL_UTILITY", "PU share"),
        (welfare_series, "Social welfare", "welfare"),
    ]

    for ax, (series, title, ylabel) in zip(axes, panels):
        xs, ys = _downsample(t, np.asarray(series, dtype=float), sample_interval)
        ax.plot(xs, ys, linewidth=1.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)

        for value, color, label in (
            (record.get("t_conv", -1), "tab:green", "converged"),
            (record.get("t_perturb_start", -1), "tab:red", "perturb start"),
            (record.get("t_perturb_end", -1), "tab:orange", "perturb end"),
            (record.get("recovery_time", -1), "tab:blue", "recovered"),
        ):
            if value is not None and int(value) > 0:
                ax.axvline(int(value), color=color, linestyle="--",
                           linewidth=1.0, alpha=0.8,
                           label=label if ax is axes[0] else None)

    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("timestep")
    fig.suptitle(
        f"seed {record.get('seed')} | gamma={record.get('gamma')} "
        f"kappa={record.get('kappa')} | "
        f"recovered={'yes' if int(record.get('recovery_time', -1)) > 0 else 'no'}",
        fontsize=11,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_file, dpi=170)
    plt.close(fig)
