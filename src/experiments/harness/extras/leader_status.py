"""
Leader and status metrics shared by Experiments B and C.

Both sweeps ask the same questions -- how large the top follower set gets, how
fast, how often leadership changes hands, and whether the leader is a STATUS
agent -- so the computation is declared once and parameterised by the two things
that actually differ:

  convergence_threshold_frac -- 0.90 in C, configurable in B
  leader_switch_margin       -- 0 in C (plain series), 1 by default in B

The margin matters more than it looks. The plain leader series breaks ties by
lowest agent id, so two agents trading a tie register a switch every step, and
kappa makes near-ties MORE common. Counting switches off the plain series
therefore conflates "leadership is unstable" with "leadership is tied", which
biases exactly the quantity a kappa sweep is trying to measure.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from experiments.harness import metrics
from experiments.harness.plugins import RunContext, RunPlugin

BASE_COLUMNS = (
    "leader_id",
    "final_top_followers",
    "time_to_90pct_followers",
    "leader_switches",
    "tail_welfare",
    "leader_role_final",
    "leader_is_status_final",
    "final_status_count",
    "tail_status_leader_share",
    "tail_status_agent_share",
)

CENSORING_COLUMNS = (
    "follower_threshold",
    "reached_follower_threshold",
    "tail_top_follower_share",
)


class LeaderStatusPlugin(RunPlugin):
    """Core leader/status metrics.

    `include_censoring` adds the three bookkeeping columns Experiment B needs to
    interpret its conditional time-to-threshold mean: which absolute threshold
    was used, whether this run reached it, and the tail follower share (which,
    unlike time-to-threshold, is defined for runs that never converge).
    """

    name = "leader_status"

    def __init__(
        self,
        *,
        threshold_frac: float = 0.90,
        leader_switch_margin: int = 0,
        include_censoring: bool = False,
        threshold_frac_arg: str | None = None,
        margin_arg: str | None = None,
    ) -> None:
        self.threshold_frac = float(threshold_frac)
        self.leader_switch_margin = int(leader_switch_margin)
        self.include_censoring = bool(include_censoring)
        self.threshold_frac_arg = threshold_frac_arg
        self.margin_arg = margin_arg
        self.columns = BASE_COLUMNS + (CENSORING_COLUMNS if include_censoring else ())

    def _frac(self, ctx: RunContext) -> float:
        if self.threshold_frac_arg:
            return float(getattr(ctx.args, self.threshold_frac_arg, self.threshold_frac))
        return self.threshold_frac

    def _margin(self, ctx: RunContext) -> int:
        if self.margin_arg:
            return int(getattr(ctx.args, self.margin_arg, self.leader_switch_margin))
        return self.leader_switch_margin

    def measure(self, ctx: RunContext) -> Dict[str, Any]:
        s = ctx.summary
        n = int(ctx.args.num_agents)

        threshold = int(np.ceil(self._frac(ctx) * (n - 1)))
        time_to_threshold = metrics.time_to_threshold(s.top_follower_series, threshold)

        # Switches are counted off the hysteretic series when a margin is set;
        # the plain series is retained for plotting continuity.
        stable_series = metrics.leader_series_hysteretic(
            s.follower_counts, margin=self._margin(ctx)
        )

        tail_welfare = (
            float(np.mean(s.social_welfare[-s.tail_window:]))
            if s.tail_window > 0
            else float("nan")
        )

        leader_role = s.final_roles[s.leader_id] if s.leader_id >= 0 else "none"

        out: Dict[str, Any] = {
            "leader_id": int(s.leader_id),
            "final_top_followers": int(max(s.final_followers)),
            "time_to_90pct_followers": int(time_to_threshold),
            "leader_switches": int(metrics.leader_switches(stable_series)),
            "tail_welfare": float(tail_welfare),
            "leader_role_final": str(leader_role),
            "leader_is_status_final": int(leader_role == "status"),
            "final_status_count": sum(1 for r in s.final_roles if r == "status"),
            "tail_status_leader_share": float(
                metrics.tail_status_leader_share(
                    s.role_history, stable_series, s.tail_window
                )
            ),
            "tail_status_agent_share": float(
                metrics.tail_status_agent_share(s.role_history, s.tail_window)
            ),
        }

        if self.include_censoring:
            out["follower_threshold"] = int(threshold)
            out["reached_follower_threshold"] = int(time_to_threshold >= 0)
            out["tail_top_follower_share"] = float(
                metrics.tail_top_follower_share(
                    s.follower_counts, s.tail_window, denom=n - 1
                )
            )

        return out
