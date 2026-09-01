"""
Shared command-line surface.

27 flags were common to all four legacy harnesses and another 6 to three of
them, each declared independently with occasionally-diverging defaults. They are
declared once here, in argument groups, with the per-experiment defaults passed
in rather than re-typed.

The parsing helpers below are byte-for-byte the semantics of the legacy
`parse_csv_ints` / `parse_kappas` / `parse_role_update_*` functions; they were
identical across the four files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence


# ---------------------------------------------------------------- parsers ---

def parse_csv_ints(text: str) -> List[int]:
    if not text or not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    if not text or not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strs(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_role_update_epochs(epoch_text: str) -> List[int]:
    if not epoch_text or not epoch_text.strip():
        return []
    parts = [p.strip() for p in epoch_text.split(",") if p.strip()]
    epochs = [int(x) for x in parts]
    return sorted(set(e for e in epochs if e > 0))


def parse_role_update_T_seq(t_text: str) -> List[int]:
    if not t_text or not t_text.strip():
        return []
    parts = [p.strip() for p in t_text.split(",") if p.strip()]
    seq = [int(x) for x in parts]
    return [t for t in seq if t > 0]


def interval_seq_from_epochs(s0: int, epochs: Sequence[int]) -> List[int]:
    prev = max(0, int(s0))
    intervals: List[int] = []
    for epoch in sorted(set(int(e) for e in epochs if int(e) > 0)):
        if epoch > prev:
            intervals.append(int(epoch - prev))
            prev = int(epoch)
    return intervals


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    """Explicit --selected-seeds wins; otherwise --seed-start plus --seeds."""
    selected = parse_csv_ints(getattr(args, "selected_seeds", "") or "")
    if selected:
        return sorted(set(seed for seed in selected if seed >= 0))
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))


# ------------------------------------------------------------ arg groups ---

def add_core_arguments(
    parser: argparse.ArgumentParser,
    *,
    defaults: Dict[str, Any] | None = None,
    steps_flag: str = "--num-steps",
) -> None:
    """The flags every experiment needs.

    `defaults` overrides any default below by dest name, so an experiment that
    genuinely wants a different default (Experiment A uses B_R=0.8/B_F=0.6 while
    B and C use 0.3/0.2) states that difference in one line instead of forking
    the whole parser.
    """
    d = dict(_CORE_DEFAULTS)
    d.update(defaults or {})

    g = parser.add_argument_group("run mode")
    g.add_argument("--mode", choices=["static", "async"], required=True)
    # Experiment D calls its horizon --num-steps-max because it stops early on
    # a convergence criterion rather than always running to the horizon.
    g.add_argument(steps_flag, type=int, default=d["num_steps"])

    g = parser.add_argument_group("population")
    g.add_argument("--num-agents", type=int, default=d["num_agents"])
    g.add_argument("--num-actions", type=int, default=d["num_actions"])

    g = parser.add_argument_group("seeds")
    g.add_argument("--seeds", type=int, default=d["seeds"], help="Number of seeds to run.")
    g.add_argument("--seed-start", type=int, default=d["seed_start"], help="First seed (inclusive).")
    g.add_argument(
        "--selected-seeds",
        type=str,
        default="",
        help='Explicit comma-separated seed list (e.g. "0,1,2"). Overrides --seeds/--seed-start.',
    )

    g = parser.add_argument_group("reward model")
    g.add_argument("--reward-base-mu", type=float, default=d["reward_base_mu"])
    g.add_argument("--reward-base-sigma", type=float, default=d["reward_base_sigma"])
    g.add_argument("--reward-agent-sigma", type=float, default=d["reward_agent_sigma"])
    g.add_argument("--reward-clip-min", type=float, default=d["reward_clip_min"])
    g.add_argument("--reward-clip-max", type=float, default=d["reward_clip_max"])

    g = parser.add_argument_group("following thresholds")
    g.add_argument("--c-threshold", type=float, default=d["c_threshold"])
    g.add_argument("--B-R", dest="B_R", type=float, default=d["B_R"])
    g.add_argument("--B-F", dest="B_F", type=float, default=d["B_F"])
    g.add_argument("--delta", type=float, default=d["delta"])

    g = parser.add_argument_group("interaction rates")
    g.add_argument("--initial-actor-rate", type=float, default=d["initial_actor_rate"])
    g.add_argument("--initial-participant-rate", type=float, default=d["initial_participant_rate"])
    g.add_argument(
        "--actor-rate-driver-mode",
        choices=["standard", "status_if_followers_kappa0"],
        default=d["actor_rate_driver_mode"],
    )
    g.add_argument(
        "--actor-rate-status-override-min-followers",
        type=int,
        default=d["actor_rate_status_override_min_followers"],
    )

    g = parser.add_argument_group("engine modes")
    g.add_argument(
        "--eq9-averaging-mode",
        choices=["participants_only", "all_agents"],
        default=d["eq9_averaging_mode"],
    )
    g.add_argument(
        "--leader-update-mode",
        choices=["participants_only_post_eq9", "all_agents_post_eq9", "participants_only_pre_eq9"],
        default=d["leader_update_mode"],
    )
    g.add_argument("--tracking-mode", choices=["full", "light"], default=d["tracking_mode"])
    g.add_argument(
        "--numpy-fast-path",
        action=argparse.BooleanOptionalAction,
        default=d["numpy_fast_path"],
    )
    g.add_argument(
        "--force-all-active-debug",
        action=argparse.BooleanOptionalAction,
        default=d["force_all_active_debug"],
    )

    add_schedule_arguments(parser, defaults=d)

    g = parser.add_argument_group("reproducibility")
    g.add_argument(
        "--seed-derivation",
        choices=["legacy_global", "direct"],
        default="legacy_global",
        help="How the engine seed is derived from the run seed. 'legacy_global' "
             "reproduces the v2/compat behaviour and is required for byte-identical "
             "parity with committed outputs; 'direct' passes the run seed straight "
             "to runtime.seed. See docs/HARNESS.md.",
    )

    g = parser.add_argument_group("analysis / output")
    g.add_argument("--tail-window", type=int, default=d["tail_window"])
    g.add_argument("--output-dir", type=str, default=d["output_dir"])


def add_schedule_arguments(
    parser: argparse.ArgumentParser,
    *,
    defaults: Dict[str, Any] | None = None,
) -> None:
    d = dict(_CORE_DEFAULTS)
    d.update(defaults or {})

    g = parser.add_argument_group("role-update schedule")
    g.add_argument("--role-update-s0", type=int, default=d["role_update_s0"])
    g.add_argument("--role-update-T-seq", type=str, default=d["role_update_T_seq"])
    g.add_argument("--role-update-base-interval", type=int, default=d["role_update_base_interval"])
    g.add_argument(
        "--fixed-role-update-interval",
        action=argparse.BooleanOptionalAction,
        default=d["fixed_role_update_interval"],
    )
    g.add_argument("--role-update-epochs", type=str, default=d["role_update_epochs"])
    g.add_argument(
        "--async-role-update-prob",
        type=float,
        default=None,
        help="Per-step Bernoulli probability for async subset role updates. "
             "If omitted, async uses independent per-agent clocks.",
    )
    g.add_argument(
        "--schedule-rng",
        choices=["global", "stream"],
        default="global",
        help="Source of randomness for async role-update timers. 'global' reproduces "
             "the legacy behaviour (np.random seeded per run) and is bit-compatible "
             "with existing outputs. 'stream' uses a dedicated Generator spawned from "
             "the run seed, which is process- and thread-safe but changes async "
             "results. See docs/HARNESS.md.",
    )


_CORE_DEFAULTS: Dict[str, Any] = {
    "num_steps": 50000,
    "num_agents": 100,
    "num_actions": 2,
    "seeds": 10,
    "seed_start": 0,
    "reward_base_mu": 0.5,
    "reward_base_sigma": 0.15,
    "reward_agent_sigma": 0.08,
    "reward_clip_min": 0.01,
    "reward_clip_max": 2.5,
    "c_threshold": 0.1,
    "B_R": 0.3,
    "B_F": 0.2,
    "delta": 1e-6,
    "initial_actor_rate": 0.7,
    "initial_participant_rate": 0.7,
    "actor_rate_driver_mode": "standard",
    "actor_rate_status_override_min_followers": 10,
    "eq9_averaging_mode": "participants_only",
    "leader_update_mode": "participants_only_post_eq9",
    "tracking_mode": "light",
    "numpy_fast_path": True,
    "force_all_active_debug": False,
    "role_update_s0": 0,
    "role_update_T_seq": "",
    "role_update_base_interval": 3000,
    "fixed_role_update_interval": True,
    "role_update_epochs": "",
    "tail_window": 500,
    "output_dir": str(Path.cwd() / "outputs"),
}
