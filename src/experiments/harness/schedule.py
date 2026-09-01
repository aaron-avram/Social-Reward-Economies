"""
Async role-update scheduling.

This loop was duplicated near-verbatim in all four legacy harnesses. It is not
experiment configuration -- it is engine scheduling semantics -- so it lives in
exactly one place and every experiment drives it through `RoleUpdateScheduler`.

Two async policies exist, selected by `--async-role-update-prob`:

  * omitted  -> independent per-agent clocks. Each agent draws an initial timer
                uniformly in [1, T_0], and on expiry advances through the
                interval sequence (T_seq, epoch-derived intervals, or a constant
                base interval).
  * given p  -> each agent updates independently with probability p per step.

RNG NOTE
--------
The legacy code drew these timers from the *global* numpy RNG (`np.random.seed`
followed by `np.random.randint`), while the engine seeds its own `RngBundle`.
Async schedules therefore depended on global interpreter state the engine does
not own: reproducible across separate processes, but silently non-deterministic
under any in-process parallelism. `rng_mode="stream"` fixes that with a
dedicated Generator spawned from the run seed. It is NOT bit-compatible with
existing async outputs, so `"global"` remains the default.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.experiments.harness.cli import (
    interval_seq_from_epochs,
    parse_role_update_epochs,
    parse_role_update_T_seq,
)


def build_async_interval_sequence(args) -> Tuple[List[int], int, str]:
    """Resolve the async interval sequence and its provenance.

    Precedence: explicit T-sequence, then epoch list, then constant base
    interval. Verbatim semantics from the legacy `_build_async_interval_sequence`.
    """
    s0 = max(0, int(args.role_update_s0))

    t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    if t_seq:
        return t_seq, s0, "T_sequence"

    epochs = parse_role_update_epochs(args.role_update_epochs)
    if epochs:
        from_epochs = interval_seq_from_epochs(s0=s0, epochs=epochs)
        if from_epochs:
            return from_epochs, s0, "epochs"

    return [max(1, int(args.role_update_base_interval))], s0, "base_interval"


class RoleUpdateScheduler:
    """Drives async role updates. A no-op in static mode.

    Usage:

        sched = RoleUpdateScheduler.build(args, seed=seed)
        for _ in range(num_steps):
            system.step()
            sched.after_step(system)
    """

    def __init__(
        self,
        *,
        mode: str,
        num_agents: int,
        interval_seq: Optional[List[int]] = None,
        role_timers: Optional[np.ndarray] = None,
        update_prob: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        refresh: bool = True,
    ) -> None:
        self.mode = mode
        self.num_agents = int(num_agents)
        self.interval_seq = interval_seq
        self.role_timers = role_timers
        self.update_prob = update_prob
        self.rng = rng
        # LEGACY INCONSISTENCY: A/B/C refreshed the last tracked state and
        # appended to role_update_times after every async role update; D's
        # `_finalize_async_step` did neither. The flag reproduces both, but the
        # difference is almost certainly an oversight in D rather than a design
        # choice -- it means D's step t does NOT reflect the post-update
        # follower graph, so its per-step follower series lags by one step
        # relative to the other three experiments.
        self.refresh = bool(refresh)
        self.interval_indices = (
            np.zeros(self.num_agents, dtype=int) if role_timers is not None else None
        )
        #: Number of role-update events applied so far. Experiment D counts
        #: these to know how many chances the population has had to recover.
        self.epoch = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, args, *, seed: int, refresh: bool = True) -> "RoleUpdateScheduler":
        mode = str(args.mode)
        n = int(args.num_agents)

        if mode != "async":
            return cls(mode=mode, num_agents=n, refresh=refresh)

        rng_mode = str(getattr(args, "schedule_rng", "global"))
        if rng_mode == "stream":
            # Sixth independent stream, spawned from the same seed the engine uses.
            (schedule_seed,) = np.random.SeedSequence(int(seed)).spawn(1)
            rng: Optional[np.random.Generator] = np.random.default_rng(schedule_seed)
        else:
            rng = None  # legacy: draw from global np.random

        prob = getattr(args, "async_role_update_prob", None)
        if prob is not None:
            return cls(mode=mode, num_agents=n, update_prob=float(prob), rng=rng,
                       refresh=refresh)

        interval_seq, async_s0, _ = build_async_interval_sequence(args)
        first_interval = int(interval_seq[0])
        if rng is None:
            role_timers = np.random.randint(1, first_interval + 1, size=n, dtype=int)
        else:
            role_timers = rng.integers(1, first_interval + 1, size=n, dtype=int)
        if async_s0 > 0:
            role_timers = role_timers + async_s0

        return cls(
            mode=mode,
            num_agents=n,
            interval_seq=list(interval_seq),
            role_timers=role_timers,
            rng=rng,
            refresh=refresh,
        )

    # -- per-step ----------------------------------------------------------

    def after_step(self, system) -> bool:
        """Apply any due role updates. Returns True if any agent was updated."""
        if self.mode != "async":
            return False
        if self.role_timers is not None:
            return self._tick_clocks(system)
        return self._tick_bernoulli(system)

    def _tick_clocks(self, system) -> bool:
        assert self.role_timers is not None and self.interval_seq is not None
        assert self.interval_indices is not None

        self.role_timers -= 1
        update_ids = np.where(self.role_timers <= 0)[0]
        if update_ids.size == 0:
            return False

        update_list = update_ids.tolist()
        self._commit(system, update_list)

        if len(self.interval_seq) == 1:
            self.role_timers[update_ids] += int(self.interval_seq[0])
        else:
            for agent_id in update_list:
                idx = int(self.interval_indices[agent_id])
                next_interval = int(
                    self.interval_seq[idx if idx < len(self.interval_seq) else -1]
                )
                self.role_timers[agent_id] += next_interval
                if idx < len(self.interval_seq) - 1:
                    self.interval_indices[agent_id] = idx + 1
        return True

    def _tick_bernoulli(self, system) -> bool:
        assert self.update_prob is not None
        if self.rng is None:
            draws = np.random.random(self.num_agents)
        else:
            draws = self.rng.random(self.num_agents)
        update_ids = np.where(draws < self.update_prob)[0]
        if update_ids.size == 0:
            return False
        self._commit(system, update_ids.tolist())
        return True

    def _commit(self, system, update_list: Sequence[int]) -> None:
        # The real engine API. `update_roles(..., refresh=True)` folds in the
        # `refresh_last_tracked_state()` that the legacy code called separately,
        # so timestep t reflects the post-update follower graph either way.
        system.update_roles(list(update_list), refresh=self.refresh)
        if self.refresh:
            system.results.role_update_times.append(int(system.time_step))
        self.epoch += 1


def async_role_interval_override(args, mode: str):
    """Schedule fields for `make_config` under async mode.

    In async mode the engine's own periodic role update must be disabled -- the
    scheduler above owns role updates entirely -- which the legacy code achieved
    by pushing the interval past the horizon. Returns
    (base_interval, s0, T_sequence, epochs).
    """
    role_interval = int(args.role_update_base_interval)
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)

    if mode == "async":
        num_steps = int(getattr(args, "num_steps", None) or getattr(args, "num_steps_max"))
        return num_steps + 1_000_000, 0, [], []

    return role_interval, role_s0, role_t_seq, role_epochs
