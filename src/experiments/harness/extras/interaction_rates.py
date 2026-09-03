"""
Actor interaction rate separation (Experiment E).

WHAT DRIVES SEPARATION
----------------------
Eq. (13) updates each agent's actor rate from a single scalar driver

    H_i = max{ J^pu_i , gamma * J^r_i , kappa * J^s_i }

so a gap in actor rates between agents is downstream of a gap in H, which is
downstream of *which term wins the max*. An agent with followers has a large
J^s, so kappa * J^s wins and pushes its rate up; an isolated agent runs on
J^pu. That means a separation measurement is uninterpretable on its own: this
plugin always records the driver decomposition alongside the rates, so a gap can
be attributed rather than merely observed.

THE CEILING PROBLEM -- READ THIS BEFORE INTERPRETING ANY RESULT
---------------------------------------------------------------
Eq. (13) has a fixed point where the two terms balance:

    exp(-(M - mu)) * u_0 = exp(-mu) * H   =>   mu* = (M + ln(H / u_0)) / 2

and mu is clipped to [0, M]. So mu saturates at M once H >= u_0 * exp(M), and at
0 once H <= u_0 * exp(-M). At the defaults (M = 1.0, u_0 = 0.1) the *entire*
dynamic range of mu corresponds to

    H in [0.0368, 0.2718]

a span of only e^2 ~ 7.4x. Outside that band every agent pins to the same
boundary and the measured separation is exactly zero no matter how different
their drivers are. mu is a log-compressed, doubly-censored readout of H.

Two consequences:

1. `share_at_ceiling` and `share_at_floor` are reported on every run, and a
   result with either near 1.0 says nothing about separation.
2. `H_leader`, `H_nonleader_mean` and friends record the *uncensored* driver, so
   you can see separation that the rate clipping has hidden.

`--M` and `--u-0` are exposed so the observable band can be moved. Raising u_0
shifts the band up; raising M widens it.

A SEPARATION MEASURE THAT DOES NOT ASSUME A LEADER
--------------------------------------------------
The question "is there separation when there is no leader" rules out any metric
defined as leader-minus-rest. Instead, sort the per-agent tail rates and find
the largest gap between consecutive values. That gap splits the population into
a top group and a bottom group with no prior assumption about who is in which,
or that either is a singleton.

To judge whether a gap is large, compare it against the null of n points spread
evenly at random over the same range, where the expected largest of the n-1
spacings is H_{n-1}/(n-1) of the range. `sep_gap_excess` is the observed gap
share divided by that null. Values near 1 mean "no more clustered than random
spread"; values well above 1 mean a real cluster boundary. This is a descriptive
reference scale, not a hypothesis test -- the null ignores that the rates are
dependent draws from a shared dynamic.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from model.agent import AgentRole

from experiments.harness.cli import parse_csv_ints
from experiments.harness.plugins import RunContext, RunPlugin

CEILING_TOL = 1e-6


# ------------------------------------------------------ separation statistics ---

def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array. 0 = perfectly equal."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0 or np.all(x <= 0):
        return 0.0
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


def expected_max_spacing_share(n: int) -> float:
    """E[largest of n-1 spacings] / range, for n points spread uniformly.

    The standard order-statistics result: H_{n-1} / (n-1). Used as the reference
    scale for `sep_gap_excess`.
    """
    if n < 3:
        return 1.0
    m = n - 1
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1, dtype=float)))
    return harmonic / m


def largest_gap_split(values: np.ndarray) -> Dict[str, float]:
    """Split a 1-D sample at its largest consecutive gap.

    Returns the gap, the gap as a share of the range, that share relative to the
    even-spread null, and the resulting group sizes and means. Makes no
    assumption that the top group is a singleton or that a leader exists.
    """
    x = np.sort(np.asarray(values, dtype=float))[::-1]  # descending
    n = x.size
    empty = {
        "sep_gap": 0.0,
        "sep_gap_range_share": 0.0,
        "sep_gap_excess": 0.0,
        "sep_top_group_size": 0,
        "sep_top_group_share": 0.0,
        "sep_top_group_mean": float("nan"),
        "sep_bottom_group_mean": float("nan"),
        "sep_threshold": float("nan"),
    }
    if n < 2:
        return empty

    gaps = x[:-1] - x[1:]
    k = int(np.argmax(gaps)) + 1          # size of the top group
    gap = float(gaps[k - 1])
    value_range = float(x[0] - x[-1])

    if value_range <= 0.0:
        return empty

    share = gap / value_range
    return {
        "sep_gap": gap,
        "sep_gap_range_share": float(share),
        "sep_gap_excess": float(share / expected_max_spacing_share(n)),
        "sep_top_group_size": int(k),
        "sep_top_group_share": float(k / n),
        "sep_top_group_mean": float(np.mean(x[:k])),
        "sep_bottom_group_mean": float(np.mean(x[k:])),
        "sep_threshold": float(0.5 * (x[k - 1] + x[k])),
    }


def rank_average(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged -- needed for a correct Spearman under ceilings.

    Saturated actor rates produce large tie blocks, and ordinal ranking would
    invent an ordering inside them.
    """
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. NaN when either input is constant."""
    if a.size < 2 or b.size != a.size:
        return float("nan")
    ra, rb = rank_average(a), rank_average(b)
    sa, sb = float(np.std(ra)), float(np.std(rb))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.mean((ra - np.mean(ra)) * (rb - np.mean(rb))) / (sa * sb))


def standardized_gap(target: float, rest: np.ndarray) -> float:
    """(target - mean(rest)) / std(rest): how many rest-SDs the target sits above.

    NaN when the rest has no spread, which is the common case under a ceiling --
    deliberately NaN rather than infinity, so it drops out of averages instead of
    poisoning them.
    """
    if rest.size < 2:
        return float("nan")
    sd = float(np.std(rest, ddof=1))
    if sd <= 0.0:
        return float("nan")
    return float((float(target) - float(np.mean(rest))) / sd)


# -------------------------------------------------------------------- plugin ---

RECORD_COLUMNS = (
    # population-level rate distribution
    "rate_mean", "rate_std", "rate_min", "rate_max", "rate_cv", "rate_gini",
    "share_at_ceiling", "share_at_floor",
    # leader-free separation
    "sep_gap", "sep_gap_range_share", "sep_gap_excess",
    "sep_top_group_size", "sep_top_group_share",
    "sep_top_group_mean", "sep_bottom_group_mean", "sep_threshold",
    "top1_rate", "top1_minus_rest_mean", "top1_z",
    # leader-conditional
    "has_leader", "leader_id", "leader_followers",
    "leader_rate", "nonleader_rate_mean", "nonleader_rate_std",
    "leader_minus_nonleader", "leader_z", "leader_rank",
    "leader_is_top1", "leader_in_top_group",
    # the uncensored driver H, and which term wins it
    "H_mean", "H_max", "H_leader", "H_nonleader_mean", "H_leader_minus_nonleader",
    "driver_share_pu", "driver_share_rep", "driver_share_status",
    "driver_share_override", "leader_driver",
    # coupling
    "spearman_rate_followers", "spearman_rate_H",
)


class ActorRateSeparationPlugin(RunPlugin):
    """Measures separation in actor interaction rates across the population.

    Rates are polled directly off the agents each step rather than read from
    `results.actor_interaction_rate_history`, so the measurement does not depend
    on `--tracking-mode`.
    """

    name = "actor_rate_separation"
    columns = RECORD_COLUMNS

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("actor-rate observation")
        g.add_argument(
            "--M", dest="M", type=float, default=1.0,
            help="Interaction budget; actor rates are clipped to [0, M]. "
                 "Together with --u-0 this sets the observable band "
                 "H in [u_0*exp(-M), u_0*exp(M)] -- outside it every agent "
                 "saturates and separation is unmeasurable.",
        )
        g.add_argument(
            "--u-0", dest="u_0", type=float, default=0.1,
            help="Outside-option utility in Eq. (13). Raises the observable "
                 "band for H; see --M.",
        )
        g.add_argument(
            "--rate-trace-seeds", type=str, default="0",
            help="Comma-separated seeds for which to emit the leader-vs-rest "
                 "actor-rate time series.",
        )
        g.add_argument(
            "--rate-trace-every", type=int, default=100,
            help="Downsample factor for the actor-rate time series.",
        )

    # -- observation -------------------------------------------------------

    def on_start(self, system, ctx: RunContext) -> None:
        ctx.scratch["rates"] = []
        ctx.scratch["followers_hist"] = []

    def on_step(self, system, ctx: RunContext, step: int, role_updated: bool) -> None:
        ctx.scratch["rates"].append(
            [float(a.state.actor_interaction_rate) for a in system.agents]
        )
        ctx.scratch["followers_hist"].append(
            [len(a.state.followers) for a in system.agents]
        )

    # -- measurement -------------------------------------------------------

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        s = ctx.summary
        args = ctx.args
        system = ctx.system

        rates_hist = np.asarray(ctx.scratch["rates"], dtype=float)   # (T, N)
        if rates_hist.size == 0:
            raise RuntimeError("no actor-rate samples recorded")

        tail = max(1, min(int(args.tail_window), rates_hist.shape[0]))
        r = rates_hist[-tail:].mean(axis=0)                          # per-agent
        n = r.size

        followers = np.asarray(s.final_followers, dtype=float)
        leader = int(s.leader_id)
        has_leader = leader >= 0

        M = float(getattr(args, "M", 1.0))
        at_ceiling = float(np.mean(r >= M - CEILING_TOL))
        at_floor = float(np.mean(r <= CEILING_TOL))

        out: Dict[str, Any] = {
            "rate_mean": float(np.mean(r)),
            "rate_std": float(np.std(r, ddof=1)) if n >= 2 else 0.0,
            "rate_min": float(np.min(r)),
            "rate_max": float(np.max(r)),
            "rate_cv": (
                float(np.std(r, ddof=1) / np.mean(r))
                if n >= 2 and np.mean(r) > 0 else float("nan")
            ),
            "rate_gini": gini(r),
            "share_at_ceiling": at_ceiling,
            "share_at_floor": at_floor,
        }
        out.update(largest_gap_split(r))

        # Top-1 versus the rest: the leaderless analogue of leader-versus-rest.
        top1 = int(np.argmax(r))
        rest = np.delete(r, top1)
        out["top1_rate"] = float(r[top1])
        out["top1_minus_rest_mean"] = (
            float(r[top1] - np.mean(rest)) if rest.size else float("nan")
        )
        out["top1_z"] = standardized_gap(r[top1], rest)

        # The uncensored driver H and which term wins the max, per agent.
        drivers = [system.agents[i].actor_rate_terms() for i in range(n)]
        H = np.array([float(d["driver"]) for d in drivers], dtype=float)
        labels = [str(d["driver_label"]) for d in drivers]
        overrides = np.array([int(d["status_override_active"]) for d in drivers])

        def _share(term: str) -> float:
            # driver_label is a pipe-joined set when terms tie exactly; count an
            # agent toward a term if that term is among the winners.
            return float(np.mean([term in lab.split("|") for lab in labels]))

        out.update({
            "H_mean": float(np.mean(H)),
            "H_max": float(np.max(H)),
            "driver_share_pu": _share("pu"),
            "driver_share_rep": _share("rep"),
            "driver_share_status": _share("status"),
            "driver_share_override": float(np.mean(overrides)),
        })

        # Leader-conditional block. Sentinels rather than omission, so the CSV
        # schema is constant and leaderless runs are explicitly visible.
        if has_leader:
            nonleader = np.delete(r, leader)
            H_nonleader = np.delete(H, leader)
            ranked = np.argsort(-r, kind="mergesort")
            out.update({
                "has_leader": 1,
                "leader_id": leader,
                "leader_followers": int(followers[leader]),
                "leader_rate": float(r[leader]),
                "nonleader_rate_mean": float(np.mean(nonleader)),
                "nonleader_rate_std": (
                    float(np.std(nonleader, ddof=1)) if nonleader.size >= 2 else 0.0
                ),
                "leader_minus_nonleader": float(r[leader] - np.mean(nonleader)),
                "leader_z": standardized_gap(r[leader], nonleader),
                "leader_rank": int(np.where(ranked == leader)[0][0]) + 1,
                "leader_is_top1": int(leader == top1),
                "leader_in_top_group": int(
                    np.where(ranked == leader)[0][0] < int(out["sep_top_group_size"])
                ),
                "H_leader": float(H[leader]),
                "H_nonleader_mean": float(np.mean(H_nonleader)),
                "H_leader_minus_nonleader": float(H[leader] - np.mean(H_nonleader)),
                "leader_driver": labels[leader],
            })
        else:
            out.update({
                "has_leader": 0,
                "leader_id": -1,
                "leader_followers": 0,
                "leader_rate": float("nan"),
                "nonleader_rate_mean": float("nan"),
                "nonleader_rate_std": float("nan"),
                "leader_minus_nonleader": float("nan"),
                "leader_z": float("nan"),
                "leader_rank": -1,
                "leader_is_top1": -1,
                "leader_in_top_group": -1,
                "H_leader": float("nan"),
                "H_nonleader_mean": float("nan"),
                "H_leader_minus_nonleader": float("nan"),
                "leader_driver": "none",
            })

        out["spearman_rate_followers"] = spearman(r, followers)
        out["spearman_rate_H"] = spearman(r, H)

        self._emit_agent_rows(ctx, r, followers, H, labels, leader)
        self._emit_timeseries(ctx, rates_hist, leader)
        return out

    # -- side tables -------------------------------------------------------

    def _emit_agent_rows(self, ctx, r, followers, H, labels, leader) -> None:
        """Per-agent tail rates: the raw material for the distribution figure."""
        base = {"mode": ctx.mode, **ctx.cell, "seed": int(ctx.seed)}
        roles = ctx.summary.final_roles
        for i in range(r.size):
            ctx.emit("agent_rates", {
                **base,
                "agent_id": int(i),
                "actor_rate": float(r[i]),
                "followers": int(followers[i]),
                "role": str(roles[i]),
                "is_leader": int(i == leader),
                "H": float(H[i]),
                "driver_label": labels[i],
            })

    def _emit_timeseries(self, ctx, rates_hist, leader) -> None:
        seeds = set(parse_csv_ints(getattr(ctx.args, "rate_trace_seeds", "") or ""))
        if int(ctx.seed) not in seeds:
            return

        every = max(1, int(getattr(ctx.args, "rate_trace_every", 100)))
        base = {"mode": ctx.mode, **ctx.cell, "seed": int(ctx.seed)}
        T = rates_hist.shape[0]

        for t in range(0, T, every):
            row = rates_hist[t]
            others = np.delete(row, leader) if leader >= 0 else row
            ctx.emit("rate_timeseries", {
                **base,
                "t": int(t + 1),
                "leader_rate": float(row[leader]) if leader >= 0 else float("nan"),
                "mean_rate": float(np.mean(row)),
                "nonleader_mean_rate": float(np.mean(others)),
                "max_rate": float(np.max(row)),
                "min_rate": float(np.min(row)),
                "p90_rate": float(np.percentile(row, 90)),
                "p10_rate": float(np.percentile(row, 10)),
            })
