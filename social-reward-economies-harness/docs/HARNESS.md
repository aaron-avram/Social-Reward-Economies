# The experiment harness

`src/experiments/harness/` is a base sweep harness; `src/experiments/v3/` holds
the four experiments as thin declarations on top of it. It replaces the four
standalone scripts in `src/experiments/v2/`, which shared roughly 60–70% of
their code by copy-paste.

Code lines, excluding blanks, comments and docstrings:

| | v2 | v3 |
|---|---|---|
| Experiment A (`pu_scaling`) | 794 | 221 |
| Experiment C (`status_scaling`) | 918 | 188 |
| Experiment B (`reputation_status_scaling`) | 1170 | 245 |
| Experiment D (`perturbation_recovery`) | 1425 | 120 |
| per-experiment subtotal | **4307** | **774** (18%) |
| shared harness | — | 2235 |
| **total** | **4307** | **3009** (70%) |

The honest number is the last row: total code drops by 30%, not by the 80% the
per-experiment column suggests. What the per-experiment column does show is that
adding or modifying an experiment now touches roughly a fifth as much code, and
that the shared 2235 lines are written once, unit tested, and fixed in one place
when wrong — none of which was true of the four copies.

## Verification

Every experiment's CSV output is **byte-identical** to its v2 counterpart for
the same arguments:

```
PYTHONPATH=src python3 tools/parity_check.py --full
```

```
exp_a_static   OK -- 5 CSV file(s) identical
exp_a_async    OK -- 3 CSV file(s) identical
exp_c_static   OK -- 3 CSV file(s) identical
exp_b_static   OK -- 3 CSV file(s) identical
exp_b_async    OK -- 3 CSV file(s) identical
exp_d_static   OK -- 4 CSV file(s) identical
exp_d_async    OK -- 4 CSV file(s) identical
```

Plus 40 unit tests over the extracted pure functions:

```
python3 -m pytest tests/harness_tests -q
```

The async cases matter most: the async role-update loop was the largest single
block of duplication, and it now exists once.

---

## Architecture

### The grid is data, not control flow

Each v2 harness hard-coded its own nested loop. `axes.py` makes the axes a
value, so one loop serves all four:

```python
Grid(axes=(Axis.of("gamma", [0, 1, 2]), Axis.of("kappa", [0, 0.1])), seeds=(0, 1))
```

* A → `(reward_model, num_states)`
* B, C → `(gamma, kappa)`
* D → `(gamma, kappa)` with one value each — a 1×1 grid swept over seeds

The cell dict then drives the CSV key columns, the `make_config` overrides, and
the aggregation grouping, uniformly. Iteration order is cells outer, seeds
inner, which is what preserves the legacy row order and makes byte-identical
parity meaningful.

Widening D from a single point to a real grid is now a one-line change.

### Two plugin lifetimes

`plugins.py` defines two protocols rather than one class with a dozen optional
methods, so it is clear which hooks fire when.

```python
class RunPlugin:                        # lives for one simulation
    columns: tuple[str, ...]            # what it contributes to the CSV
    def add_arguments(self, parser)     # CLI flags
    def configure(self, config, ctx)    # amend SystemConfig before construction
    def on_start(self, system, ctx)
    def before_step(self, system, ctx, next_step)   # interventions land here
    def on_step(self, system, ctx, step, role_updated)
    def on_finish(self, system, ctx)
    def measure(self, ctx) -> dict      # must return exactly `columns`

class SweepPlugin:                      # lives for the whole grid
    def add_arguments(self, parser)
    def figures(self, ctx)              # plots and extra CSVs
```

Most experiments only declare metrics, so `MetricsPlugin` adapts a plain
function:

```python
MetricsPlugin("core", CORE_COLUMNS, measure_core)
```

Experiment D is the design's stress test and the reason `before_step` exists: it
is a state machine (converge → perturb → recover) that must modify the leader's
policy *before* the engine steps. If the protocol can express D it can express
anything the other three need.

### Column ownership is checked at startup

`Experiment.record_columns` is the authoritative CSV contract. At construction
the harness checks that no two plugins claim the same column and that the union
of plugin columns exactly equals the schema. A metric computed but never
written, or a column declared and never filled, is an error before the first
simulation runs rather than a column that silently vanishes from a six-hour
sweep.

### Aggregation is declared, not spelled out

The v2 `aggregate()` functions were 60–90 lines of `m1, s1, c1 = ...` feeding a
positional dataclass — a form where inserting a metric in the middle silently
renames every column after it. Now:

```python
aggregate_spec=(
    Derived("reach_rate", _reach_rate),
    Triple("final_top_followers"),
    Triple("time_to_90pct_followers", where=nonneg, fallback=[-1.0]),
)
```

`Triple` emits `mean_/std_/ci95_`; `Derived` emits one computed column.
Filtering is explicit at each site, which is how it should be — several of these
metrics use sentinel values that must not be averaged in.

### What moved out of the experiments

* **`schedule.py`** — the async role-update loop, previously duplicated four
  times. This is engine scheduling, not experiment configuration.
* **`metrics.py`** — leader series, switch counting, time-to-threshold, tail
  shares, mean/std/CI. Pure functions of arrays, now unit tested.
* **`extras/norm_optimality.py`** — the brute-force norm search, previously
  identical in B and C.
* **`cli.py`** — 27 flags common to all four, 6 more common to three, declared
  once with per-experiment defaults passed in.
* **`configspec.py`** — one `make_config` with an explicit override dict.

---

## Findings

Five things surfaced during the port. The first two change how you should think
about the existing results.

### 1. The v2 outputs cannot be reproduced from the clean engine API

`compat.py`'s `MultiAgentSystem.__init__` does this whenever
`runtime.seed == 0` — which is always, because no v2 harness ever passed a seed:

```python
derived = int(np.random.randint(0, 2**31 - 1))
config = replace(config, runtime=replace(config.runtime, seed=derived))
```

The engine seed is drawn from the *global* stream that `np.random.seed(seed)`
just seeded. Trajectories are a deterministic function of the run seed, but
through a layer of laundering that exists only inside the shim. Every committed
CSV in `outputs/` depends on it.

`--seed-derivation` exposes both:

* `legacy_global` (default) — reproduces v2 exactly. Required for parity.
* `direct` — `runtime.seed = s`, what the engine API intends and what the unit
  tests assume. Different, equally valid trajectories.

**Consequence for the migration plan:** deleting `compat.py` invalidates the
committed results unless this flag is kept. `compat.py`'s own docstring says
output comparisons "must be distributional" — it is right, and this is the
mechanism.

### 2. Experiment D was broken against the refactored engine

`v2/perturbation_recovery.py` line 550 called `system.compute_observer_utility`,
which does not exist on the new engine and which `compat.py` does not shim. The
script raised `AttributeError` on the first perturbation step. Its committed
outputs predate the engine refactor and have not been regenerated since.

Fixed in place (one line, to `system.rewards.observer_utility`) so the parity
comparison could run. Worth confirming whether anything downstream depends on
those stale D outputs.

Two adjacent references in the same file are also suspect and were ported to the
real API: `system._shared_good_actions` (now on `system.rewards`) and
`system._s_matrix` (now `system.rep.s`).

### 3. Async role updates depend on global RNG state

Async timers are drawn from `np.random` while the engine uses its own
`RngBundle`. Reproducible across separate processes, silently non-deterministic
under any in-process parallelism. `--schedule-rng stream` spawns a dedicated
`Generator` from the run seed; it is not bit-compatible with existing async
outputs, so `global` remains the default.

### 4. Two silent behavioural divergences between experiments

Both preserved, both flagged where they live:

* **D's async path never refreshed the tracked state** after a role update,
  while A/B/C did. D's step *t* therefore does not reflect the post-update
  follower graph — its follower series lags the other three by one step. Almost
  certainly an oversight. Reproduced via `async_refresh=False`.
* **D's local `_mean_std_ci` dropped non-finite values**; A/B/C's did not, and
  filtered explicitly per call site instead. Reproduced via `finite_only=True`.

### 5. Experiment C's threshold flags did nothing

`v2/status_scaling.py` declared `--c-threshold`, `--B-R` and `--B-F` and then
hardcoded `0.1 / 0.3 / 0.2` in `make_config`. Reproduced rather than fixed —
fixing it would invalidate every committed C figure — but now stated in one
visible place, with `--respect-threshold-flags` to opt into the correct
behaviour. B had already fixed this independently.

---

## Statistics caveats

Two things worth knowing before reading any figure the harness produces.

**The confidence intervals are normal approximations**, `1.96 · s/√n`, not
t-intervals. With the 3–10 seeds these sweeps typically use, that understates
the true 95% width by roughly 10–25%. Preserved for consistency with published
figures; treat the bars as indicative.

**`mean_time_to_90pct_followers` is conditional on reaching the threshold**, and
whether a run reaches is itself a function of γ and κ. Read it alongside
`reach_rate` (Experiment B) — the mean alone will suggest high-κ cells converge
*faster* when in fact most of them never converge at all. The same applies to
Experiment D's `mean_recovery_time` and `recovery_rate`.

---

## Usage

```bash
# Experiment A: personal-utility baseline
PYTHONPATH=src python3 src/experiments/v3/exp_a_pu_scaling.py \
    --mode static --num-states-list 5,10,20 --seeds 10

# Experiment C: status scaling
PYTHONPATH=src python3 src/experiments/v3/exp_c_status_scaling.py \
    --mode static --gammas 0,2,4 --kappas 0,0.05,0.1 --seeds 10

# Experiment B: full gamma x kappa grid
PYTHONPATH=src python3 src/experiments/v3/exp_b_reputation_status_scaling.py \
    --mode async --gammas 0,1,2,3,4 --kappas 0,0.01,0.05 --kappa-scale-by-n

# Experiment D: perturbation and recovery
PYTHONPATH=src python3 src/experiments/v3/exp_d_perturbation_recovery.py \
    --mode static --gamma 5 --kappa 2 --seeds 3
```

## Adding an experiment

```python
EXPERIMENT = Experiment(
    name="my_sweep",
    description="...",
    build_parser=build_parser,        # cli.add_core_arguments + your flags
    build_axes=build_axes,            # args -> tuple[Axis, ...]
    run_plugins=(MetricsPlugin("core", CORE_COLUMNS, measure_core),),
    sweep_plugins=(MyFigures(),),
    record_columns=("mode", *axis_names, "seed", *CORE_COLUMNS),
    aggregate_spec=(Triple("some_metric"),),
)

if __name__ == "__main__":
    EXPERIMENT.main()
```

Reuse `LeaderStatusPlugin`, `NormOptimalityPlugin`, `ConsensusTrackerPlugin`,
`ActorRateTrackerPlugin`, `FollowerProgressionPlugin` and `AgentTracePlugin`
from `harness/extras/` before writing new measurement code.

---

## Remaining work

1. **Move `schedule.py` into `model/`.** It is engine scheduling and belongs
   next to `ScheduleParams`, which currently exists and is half-used. Kept in
   `harness/` here so this change touched no engine file and broke no parity
   test.
2. **Stop the engine printing during `step()`.** Every v2 harness wrapped its
   loop in `redirect_stdout(io.StringIO())`. That workaround now lives once, in
   `runner.silence_engine`, but it is a workaround: progress belongs on the
   `Recorder` / `progress_printer` path.
3. **Delete `compat.py` and `experiments/v1/`.** Blocked on deciding what to do
   about finding 1 — either keep `--seed-derivation legacy_global` permanently,
   or regenerate the results under `direct` and retire the old CSVs.
4. **Move `outputs/` out of `src/`.** 17 MB of PNGs and CSVs are committed with
   no ignore rule. `results/` at the repo root, gitignored, with the existing
   parameter-stamped subdirectory naming.
5. **Switch async runs to `--schedule-rng stream`** once you are ready to accept
   a one-time change in async outputs. This is a prerequisite for any in-process
   parallelism.
