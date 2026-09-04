# Social Reward Economies

Multi-agent simulations of norm emergence under reputation and status incentives.

Agents repeatedly choose actions in a state-dependent environment. Each one optimises
one of three objectives — its own payoff (**PU**), its reputation among others (**REP**),
or its status, i.e. the size of its follower set (**STATUS**) — and periodically
re-selects that objective. Reputation spreads by gossip; agents follow whoever they
believe is most reputable, and a follower adopts that agent's policy. The question the
experiments ask is when this produces a single *opinion leader* whose policy becomes a
shared norm, how fast, how stably, and whether the resulting norm is welfare-optimal.

The model follows *Learning Common Norms in Multi-Agent Systems* (Vedic Sharma and
Peter Marbach). Equation numbers in the code (Eq. 9, Eq. 13) refer to that paper.

This repository is a rebuild of
[jennifer1046/social_reward_economies](https://github.com/jennifer1046/social_reward_economies).
That repository's simulator was a single 2,676-line module (`src/code_debugged.py`) with
four standalone experiment scripts of ~1,000–1,700 lines each. This one decomposes the
simulator into a package and the experiment scripts into declarations over a shared
harness, with the originals kept in-tree as an executable reference. See
[Lineage](#lineage) for exactly what was preserved and what changed.

---

## Contents

- [Requirements and setup](#requirements-and-setup)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Architecture](#architecture)
- [The experiments](#the-experiments) — [A](#experiment-a--personal-utility-baseline) · [B](#experiment-b--reputation--status-grid) · [C](#experiment-c--status-scaling) · [D](#experiment-d--perturbation-and-recovery) · [E](#experiment-e--actor-rate-separation)
- [Shared CLI reference](#shared-cli-reference)
- [Output files](#output-files)
- [Reading the results](#reading-the-results)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Lineage](#lineage)
- [Known issues](#known-issues)

---

## Requirements and setup

Python **3.9+** (`argparse.BooleanOptionalAction` and PEP 585 generics are used
unguarded; `pyproject.toml` claims 3.8, which is wrong — see
[Known issues](#known-issues)). Dependencies: `numpy`, `matplotlib`, `pandas`,
`pytest`.

```bash
git clone https://github.com/aaron-avram/Social-Reward-Economies.git
cd Social-Reward-Economies

python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

The editable install puts `src/` on the path, so `model` and `experiments` import
cleanly. Without it, prefix every command with `PYTHONPATH=src`; that is the form used
throughout this README so the commands work either way.

Every experiment script writes PNGs, so a headless machine wants
`export MPLBACKEND=Agg`.

## Quick start

```bash
# Fast checks: unit + harness tests, a few seconds.
PYTHONPATH=src python3 -m pytest -q tests/unit_tests tests/harness_tests

# A small end-to-end sweep (~2 s).
PYTHONPATH=src python3 src/experiments/v3/exp_a_pu_scaling.py \
    --mode static --num-agents 8 --num-states-list 3 \
    --num-steps 400 --seeds 2 --role-update-base-interval 100 \
    --output-dir /tmp/srewards/expA

# Confirm the v3 ports still match the v2 harnesses byte-for-byte (~2 min).
PYTHONPATH=src python3 tools/parity_check.py
```

## Repository layout

```
src/
  model/                     the simulation engine, one concern per module
    system.py                MultiAgentSystem — composition root, per-step phases
    agent.py                 agent state, roles, policy
    reputation.py            gossip, reputation learning, follower formation (Phase 4)
    roleupdate.py            periodic PU/REP/STATUS role reassignment
    rewards.py               the four reward models
    welfare.py               true reputation, opinion leader, paper welfare
    results.py               SimulationResults / StepRecord
    instrumentation.py       Recorder protocol: Null / Full, dense-history export
    config.py                nested frozen config dataclasses
    plots.py, rng.py

  experiments/
    harness/                 the shared experiment runtime
      experiment.py          the Experiment declaration and its main()
      runner.py              run_single — the one simulation loop
      cli.py                 shared flags and parsers
      configspec.py          args + grid cell -> SystemConfig
      axes.py                sweep axes and the grid
      plugins.py             RunPlugin / SweepPlugin, column-ownership check
      aggregate.py           Triple / Derived aggregation spec
      metrics.py, plotting.py, schedule.py, tables.py
      extras/                per-experiment measurement plugins
        leader_status.py     leader/status metrics shared by B and C
        norm_optimality.py   final vs best norm welfare
        tracking.py          consensus, actor rates, follower progression, agent traces
        perturbation.py      Experiment D's state machine (894 lines)
        interaction_rates.py Experiment E's separation statistics

    v3/                      >>> current experiments <<<
      exp_a_pu_scaling.py
      exp_b_reputation_status_scaling.py
      exp_c_status_scaling.py
      exp_d_perturbation_recovery.py
      exp_e_actor_rate_separation.py

    v2/                      frozen legacy harnesses on the new engine (via compat.py)
    v1/                      frozen legacy harnesses on the frozen engine
    v2/outputs/              committed results and report figures

  benchmark_code.py          the frozen pre-refactor engine, kept for parity tests

tests/
  unit_tests/                per-module tests of model/            (~3 s)
  harness_tests/             harness plumbing                      (~1 s)
  parity_tests/              model/ vs benchmark_code.py           (~35 s)
  experiment_tests/          v1 vs v2 distributional comparison    (~2 min)

tools/
  parity_check.py            differential test: v2 harness vs v3 port, CSV by CSV
```

## Architecture

### The engine (`src/model/`)

`MultiAgentSystem` is the composition root: it holds the whole `SystemConfig` and the
whole `RngBundle`, and everything below it receives narrow slices. One `step()` runs
these phases:

1. **Sample active sets.** Each agent independently becomes an actor or a participant
   according to its interaction rates.
2. **Actors act.** Sample actions from the current policies, draw payoffs from the
   reward model.
3. **Role-based updates.** PU agents update their policy toward payoff; STATUS agents
   toward follower count; REP agents toward reputation.
4. **Reputation learning** (`reputation.phase4`). Gossip mixes reputation estimates,
   Eq. 9 averages the observed utilities, followers form and dissolve against the
   `B_R` / `B_F` hysteresis band, and the opinion leader is recomputed.
5. **Adopt leader behaviour.** Followers copy their leader's policy.
6. **Role update** (periodically, on the schedule) — agents re-choose PU / REP / STATUS.

Configuration is nested and frozen: `Dimensions`, `AlgorithmParams`, `RewardParams`,
`RuntimeParams`, `ScheduleParams`, `StepsizeParams`. `AlgorithmParams.__post_init__`
enforces `0 ≤ B_F < B_R ≤ 1`, `c_threshold ∈ [0,1]`, `γ, κ ≥ 0`, so a malformed sweep
fails at construction rather than 40,000 steps in.

The two weights that the experiments sweep:

| Symbol | Flag / axis | Meaning |
| --- | --- | --- |
| γ (gamma) | `--gammas` | weight on the reputation term in an agent's objective |
| κ (kappa) | `--kappas` | weight on the status (follower-count) term |

An agent's interaction-rate driver is `H_i = max{J^pu_i, γ·J^r_i, κ·J^s_i}`; which term
wins determines what the agent is optimising and, through Eq. 13, how much it interacts.

### The harness (`src/experiments/harness/`)

An experiment is a **value**, not a script. Each v3 file constructs one `Experiment`
dataclass:

```python
EXPERIMENT = Experiment(
    name="status_scaling",              # filename prefix for every CSV and PNG
    build_parser=build_parser,          # argparse setup
    build_axes=build_axes,              # args -> sweep axes
    run_plugins=(...),                  # per-run measurement / intervention
    sweep_plugins=(...),                # figures over the finished grid
    record_columns=(...),               # the CSV contract
    aggregate_spec=(...),               # what to aggregate and how
    config_overrides={...},             # fields this experiment pins
)
```

`Experiment.main()` is the same driver for all five. It builds the grid, runs every
(cell, seed) through `run_single`, collects each plugin's declared columns into a
record, aggregates, writes CSVs, and calls the sweep plugins for figures.

`record_columns` looks redundant next to the plugins that populate it, and that is the
point: it is checked against plugin column ownership at startup. A metric no plugin
claims, or a plugin column missing from the schema, is a startup error rather than a
column that silently vanishes from a sweep that took six hours.

**Plugins** hook the run at `add_arguments`, `configure` (amend the `SystemConfig`),
`on_start`, `before_step` (interventions), `on_step` (observation), `on_finish`, and
`measure` (emit columns). Experiment D's whole perturbation state machine is one
`RunPlugin`; Experiment B's κ̃ → κ/N rescaling is a three-line `configure`.

**Aggregation** is declarative. `Triple("x")` emits `mean_x`, `std_x`, `ci95_x`, with
optional `where=` filters (`nonneg`, `positive`, `finite`) and `finite_only=`.
`Derived("name", fn)` computes anything else over the group — this is where the reach
rates and conditional means live.

## The experiments

Every experiment takes the same core flags (see
[Shared CLI reference](#shared-cli-reference)); `--mode {static,async}` is **required**
for all five, including D. Each section below gives:

- a **test command** — a few seconds, small N, short horizon, for checking the pipeline
  end to end;
- an **actual command** — the production settings, matched to the committed outputs in
  `src/experiments/v2/outputs/` where those exist.

Runtimes below were measured in a Linux container and scale roughly linearly in
`steps × seeds × grid cells`; at N = 100 the engine costs about 4 s per 1,000 steps.

---

### Experiment A — personal-utility baseline

`src/experiments/v3/exp_a_pu_scaling.py`

**The control.** γ = 0 and κ = 0 are pinned in `config_overrides`, so social incentives
are switched off entirely and every agent optimises only its own payoff. Nothing should
follow anyone: the expected result is `final_top_followers == 0` and `leader_id == -1`
in every run. Sweeps `num_states` and `reward_model`.

**Sweep axes:** `--num-states-list`, `--reward-models` (reward model is the outer loop,
which fixes CSV row order).

**Test command** (~2 s):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_a_pu_scaling.py \
    --mode static \
    --num-agents 8 \
    --num-states-list 3,4 \
    --num-steps 400 \
    --seeds 3 \
    --role-update-base-interval 100 \
    --trace-seeds 0 --trace-every 50 \
    --output-dir /tmp/srewards/expA_test
```

**Actual command** — reproduces `outputs/exp_a/final_pu_baseline/` (~10 min):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_a_pu_scaling.py \
    --mode static \
    --num-agents 100 \
    --num-states-list 10 \
    --num-steps 50000 \
    --seeds 3 \
    --reward-models shared_base_gaussian \
    --role-update-base-interval 3000 \
    --trace-seeds 0,1,2 --trace-every 100 \
    --output-dir src/experiments/v3/outputs/exp_a/final_pu_baseline
```

Add `--mode async` for the asynchronous role-update schedule; the two modes write
separate files (`..._static.csv` vs `..._async.csv`) and do not collide.

**Outputs:** `pu_scaling_runs_*.csv`, `pu_scaling_aggregate_*.csv`,
`pu_scaling_seed_comparison_*.csv`, `pu_progression_*.csv`, `pu_agent_traces_*.csv`,
nine metric PNGs, a per-configuration progression PNG, and the Section 5.1 report figure
`expA_followers_timeseries.png`.

---

### Experiment B — reputation × status grid

`src/experiments/v3/exp_b_reputation_status_scaling.py`

The full **γ × κ grid**: how the status weight affects the stability and the speed of
convergence to an opinion leader, at each level of reputation weight. This is the
experiment with the most measurement machinery — leader/status metrics, norm
optimality, consensus episodes, actor-rate traces, and explicit censoring bookkeeping.

**Sweep axes:** `--gammas` × `--kappas` (full cross product).

**Flags specific to B:**

| Flag | Default | Effect |
| --- | --- | --- |
| `--kappa-scale-by-n` | off | interpret `--kappas` as κ̃ and pass κ = κ̃/N to the engine, so the status term does not swamp reputation at realistic N. The CSV keeps the swept κ̃, so figures stay labelled in κ̃. |
| `--leader-switch-margin` | `1` | followers by which a challenger must *strictly exceed* the incumbent before a switch is counted. `0` reproduces lowest-agent-id tie-breaking, which inflates `leader_switches` under the near-ties that κ makes common. |
| `--convergence-threshold-frac` | `0.90` | follower fraction of (N−1) defining convergence. |

**Test command** (~4 s):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_b_reputation_status_scaling.py \
    --mode static \
    --num-agents 8 --num-states 3 \
    --num-steps 400 --seeds 3 \
    --gammas 0,2 --kappas 0,1 \
    --role-update-base-interval 100 \
    --output-dir /tmp/srewards/expB_test
```

**Actual command** — reproduces `outputs/exp_b/gamma_sweep_seed0/` (~5 min):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_b_reputation_status_scaling.py \
    --mode static \
    --num-agents 100 --num-states 3 \
    --num-steps 10000 \
    --selected-seeds 0 \
    --gammas 4,5 \
    --kappas 0,0.4,0.8 \
    --reward-model simple_preferred_action \
    --role-update-base-interval 3000 \
    --output-dir src/experiments/v3/outputs/exp_b/gamma_sweep_seed0
```

A fuller grid with proper seed replication (~1.5 h; this is the shape you want for a
publishable cell estimate):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_b_reputation_status_scaling.py \
    --mode static \
    --num-agents 100 --num-states 3 \
    --num-steps 10000 --seeds 20 \
    --gammas 0,1,2,3,4,5 \
    --kappas 0,0.1,0.2,0.4,0.8 \
    --kappa-scale-by-n \
    --leader-switch-margin 1 \
    --role-update-base-interval 3000 \
    --output-dir src/experiments/v3/outputs/exp_b/full_grid
```

**Outputs:** runs / aggregate / seed-comparison CSVs, six γ×κ heatmaps, and five
error-bar report figures written to `<output-dir>/../final_report_figures/`.

> **Reading `mean_time_to_90pct_followers`:** it is conditional on *reaching* the
> threshold, and whether a run reaches is itself a function of γ and κ. Always read it
> next to `reach_rate`. The mean alone will suggest that high-κ cells converge faster
> when in fact most of them never converge at all. `n_reached` gives the count behind
> each cell, and `tail_top_follower_share` is defined even for runs that never converge.

---

### Experiment C — status scaling

`src/experiments/v3/exp_c_status_scaling.py`

Sweeps γ × κ and asks whether raising the status weight makes the emergent opinion
leader a STATUS agent, and what that does to welfare and to the optimality of the norm
the leader ends up broadcasting. Lighter than B: leader/status metrics plus norm
optimality, no consensus or actor-rate tracking.

**Behaviour preserved deliberately.** The v2 harness declared `--c-threshold`, `--B-R`
and `--B-F` on the CLI and then hardcoded `0.1 / 0.3 / 0.2`, so those three flags did
nothing. The port reproduces that rather than silently fixing it, because fixing it
would invalidate every committed Experiment C figure — but now it is stated in one
visible place instead of buried in a 55-line function. Pass **`--respect-threshold-flags`**
to opt into the corrected behaviour; results will differ.

`LeaderStatusPlugin` runs with `leader_switch_margin=0` here, matching v2. Compare
against B, which defaults to `1`.

**Test command** (~4 s):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_c_status_scaling.py \
    --mode static \
    --num-agents 8 --num-states 3 \
    --num-steps 400 --seeds 3 \
    --gammas 0,2 --kappas 0,1 \
    --role-update-base-interval 100 \
    --output-dir /tmp/srewards/expC_test
```

**Actual command** — reproduces `outputs/exp_c/final_status_sweep/` (~5 min):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_c_status_scaling.py \
    --mode static \
    --num-agents 100 --num-states 3 \
    --num-steps 10000 \
    --selected-seeds 0 \
    --gammas 5 \
    --kappas 0,0.005,0.01,0.015,0.02,0.03,0.05 \
    --reward-model simple_preferred_action \
    --role-update-base-interval 3000 \
    --output-dir src/experiments/v3/outputs/exp_c/final_status_sweep
```

Note that the committed sweep is a **single γ and a single seed**. Heatmaps are only
emitted when both axes have more than one value, so that run produces line figures
only. For the 2-D picture with replication (~2 h):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_c_status_scaling.py \
    --mode static \
    --num-agents 100 --num-states 3 \
    --num-steps 10000 --seeds 20 \
    --gammas 0,1,2,3,4,5 \
    --kappas 0,0.005,0.01,0.02,0.05,0.1 \
    --role-update-base-interval 3000 \
    --output-dir src/experiments/v3/outputs/exp_c/full_grid
```

---

### Experiment D — perturbation and recovery

`src/experiments/v3/exp_d_perturbation_recovery.py`

Run until an opinion leader emerges, force that leader to behave badly for a fixed
window, then measure whether and how leadership recovers — and whether the leader who
comes back is the same one. The state machine lives in
`harness/extras/perturbation.py`; the experiment file is just the declaration.

D sweeps **only seeds**: `--gamma` and `--kappa` are scalars, declared as single-valued
axes so the CSV key columns and aggregate grouping come out right with no special
casing. Widening it later is a one-line change. It uses `--num-steps-max` rather than
`--num-steps` because it stops on a convergence criterion rather than always running to
the horizon.

**Flags specific to D:**

| Group | Flags |
| --- | --- |
| Perturbation | `--perturb-strength` (8.0), `--perturb-duration` (600), `--perturb-policy-mode {targeted_low_payoff,force_bad_action}`, `--collapse-followers-on-perturb`, `--reputation-shock-factor` (1.0), `--post-window` (2500) |
| Criteria | `--conv-threshold`, `--conv-hold-steps` (200), `--recovery-threshold` (0.9), `--recovery-hold-steps` (150), `--stable-tail-window` (200), `--dominant-threshold` (0.5), `--drop-fraction-threshold` (0.5) |
| Output | `--output-prefix`, `--run-label`, `--auto-run-subdir` / `--no-auto-run-subdir` |

Thresholds ≤ 1 are read as ratios of (N−1); above 1 as absolute follower counts.

`--collapse-followers-on-perturb` changes what is being measured: dissolving the
leader's follower set at perturbation start turns the experiment into
re-formation-from-scratch rather than resilience. Leave it off unless that is the
question.

By default D writes into a parameter-stamped subdirectory
(`perturbation_recovery_static_g5_k2_N50_S7_steps44000_seed0to2_<timestamp>/`) so two
runs with different settings cannot overwrite each other. `--no-auto-run-subdir` writes
straight into `--output-dir`, which is what `tools/parity_check.py` uses.

**Test command** (~3 s):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_d_perturbation_recovery.py \
    --mode static \
    --num-agents 8 --num-states 3 \
    --num-steps-max 1200 --seeds 2 \
    --gamma 5 --kappa 0 \
    --role-update-base-interval 60 \
    --conv-threshold 0.5 --conv-hold-steps 5 \
    --recovery-threshold 0.4 --recovery-hold-steps 5 \
    --perturb-duration 50 --post-window 100 \
    --no-auto-run-subdir \
    --output-dir /tmp/srewards/expD_test
```

**Actual command** — reproduces `outputs/exp_d/observational_baseline/` (~10 min):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_d_perturbation_recovery.py \
    --mode static \
    --num-agents 50 --num-states 7 \
    --num-steps-max 44000 \
    --seeds 3 \
    --gamma 5 --kappa 2 \
    --reward-model simple_preferred_action \
    --role-update-base-interval 3000 \
    --perturb-duration 600 --post-window 2500 \
    --output-dir src/experiments/v3/outputs/exp_d/observational_baseline
```

**Outputs:** runs / aggregate CSVs, a per-seed trajectory PNG, and a per-seed
`*_exit_diagnostics.csv` explaining why each run left each phase.

> **Reading `mean_recovery_time`:** conditional on recovering at all. The aggregate
> spec puts the five rates first for exactly this reason — `conv_rate`, `drop_rate`,
> `normless_rate`, `recovery_rate`, `stable_recovery_rate`. Read every conditional mean
> against its rate. In the committed baseline, two of three seeds have
> `recovery_time == -1`; averaging the third alone would report a recovery time for a
> setting that mostly does not recover.

---

### Experiment E — actor-rate separation

`src/experiments/v3/exp_e_actor_rate_separation.py`

The newest experiment, with no v2 counterpart and no committed outputs. It sweeps γ × κ
and asks two things:

1. When an opinion leader emerges, does its **actor interaction rate** separate from
   everyone else's?
2. When no leader emerges, is there any separation at all — a top group that is not a
   leader, or none?

Both are answered by the same measurements, because the separation statistic
(`sep_gap_excess`, from the largest gap in the sorted per-agent rates) never assumes a
leader exists. Leader-conditional columns sit alongside it, so leaderless runs are
visible rather than silently dropped. This is the simulation side of the theoretical
question of why an opinion leader's actor rate should exceed other agents'.

**Sweep axes:** `--gammas` × `--kappas`, plus an optional third axis
`--initial-actor-rates`. That third axis exists because Eq. 13 is a gradient flow toward
μ\* : if every agent starts at the same rate and that rate is already at the ceiling, no
separation can ever appear, and a null result would be an artefact of the initial
condition rather than a property of the dynamics.

**Flags specific to E:** `--M` (interaction budget, default 1.0), `--u-0` (outside-option
utility, default 0.1), `--rate-trace-seeds`, `--rate-trace-every`.

**Test command** (~6 s):

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_e_actor_rate_separation.py \
    --mode static \
    --num-agents 10 --num-states 3 \
    --num-steps 400 --seeds 2 \
    --gammas 0,2 --kappas 0,1 \
    --role-update-base-interval 100 \
    --rate-trace-seeds 0 --rate-trace-every 50 \
    --output-dir /tmp/srewards/expE_test
```

**Actual command** (~1 h). Note that `--num-states 7` and `--num-actions 3` are passed
explicitly: the file's intended defaults for those two are not applied because of a
key-naming bug (see [Known issues](#known-issues)).

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_e_actor_rate_separation.py \
    --mode static \
    --num-agents 50 --num-states 7 --num-actions 3 \
    --num-steps 10000 --seeds 10 \
    --gammas 0,1,2,3,4 \
    --kappas 0,0.01,0.02,0.05,0.1 \
    --M 1.0 --u-0 0.1 \
    --role-update-base-interval 3000 \
    --rate-trace-seeds 0,1 --rate-trace-every 100 \
    --output-dir src/experiments/v3/outputs/exp_e/separation_grid
```

With the initial-condition axis, to rule out a ceiling artefact:

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 src/experiments/v3/exp_e_actor_rate_separation.py \
    --mode static \
    --num-agents 50 --num-states 7 --num-actions 3 \
    --num-steps 10000 --seeds 10 \
    --gammas 0,2,4 --kappas 0,0.02,0.1 \
    --initial-actor-rates 0.2,0.5,0.7 \
    --u-0 0.3 \
    --role-update-base-interval 3000 \
    --output-dir src/experiments/v3/outputs/exp_e/initial_condition_sweep
```

> **Read `harness/extras/interaction_rates.py` before interpreting any of this.** Two
> things dominate:
>
> - Actor rates are clipped to `[0, M]`, and Eq. 13's fixed point is
>   μ\* = (M + ln(H/u₀))/2. At the defaults (M = 1, u₀ = 0.1) the whole observable range
>   of μ covers only H ∈ [0.037, 0.272]. Outside that band every agent pins to the same
>   boundary and measured separation is zero **regardless of the true driver gap**.
>   Check `mean_share_at_ceiling` first — near 1.0 means the run says nothing about
>   separation, and you should raise `--u-0` (or `--M`) and rerun. The `H_*` columns
>   record the uncensored driver for exactly this reason.
> - Separation in rates is downstream of which term wins in
>   H_i = max{J^pu, γ·J^r, κ·J^s}. `driver_share_status` tells you whether the status
>   term is winning for anyone at all. If it is zero, no amount of follower structure
>   will separate the rates, because κ·J^s never enters the max.
>
> The figure that answers the headline question is
> `figures/expE_separation_leader_vs_leaderless.png`: it plots
> `mean_sep_gap_excess_with_leader` against `mean_sep_gap_excess_no_leader` on shared
> axes, holding γ and κ fixed, so a difference cannot be attributed to the sweep
> parameters. The horizontal line at 1.0 is the even-spread null.

---

## Shared CLI reference

Declared once in `harness/cli.py`, with per-experiment defaults passed in rather than
re-typed. Defaults below are the harness defaults; individual experiments override
some (Experiment A uses `B_R=0.8 / B_F=0.6` where B and C differ, and so on) — run
`--help` on a script for its effective values.

| Group | Flags | Default |
| --- | --- | --- |
| Run mode | `--mode {static,async}` | **required** |
| | `--num-steps` (`--num-steps-max` in D) | 50000 |
| Population | `--num-agents`, `--num-actions` | 100, 2 |
| Seeds | `--seeds`, `--seed-start`, `--selected-seeds` | 10, 0, — |
| Reward model | `--reward-base-mu`, `--reward-base-sigma`, `--reward-agent-sigma`, `--reward-clip-min`, `--reward-clip-max` | 0.5, 0.15, 0.08, 0.01, 2.5 |
| Following | `--c-threshold`, `--B-R`, `--B-F`, `--delta` | 0.1, 0.3, 0.2, 1e-6 |
| Interaction rates | `--initial-actor-rate`, `--initial-participant-rate`, `--actor-rate-driver-mode`, `--actor-rate-status-override-min-followers` | 0.7, 0.7, standard, 10 |
| Engine modes | `--eq9-averaging-mode`, `--leader-update-mode`, `--tracking-mode {full,light}`, `--numpy-fast-path`, `--force-all-active-debug` | participants_only, participants_only_post_eq9, light, on, off |
| Role-update schedule | `--role-update-s0`, `--role-update-T-seq`, `--role-update-base-interval`, `--fixed-role-update-interval`, `--role-update-epochs`, `--async-role-update-prob`, `--schedule-rng {global,stream}` | 0, —, 3000, on, —, —, global |
| Reproducibility | `--seed-derivation {legacy_global,direct}` | legacy_global |
| Output | `--tail-window`, `--output-dir` | 500, `<script dir>/outputs` |

`--selected-seeds "0,3,7"` overrides `--seeds` / `--seed-start` entirely.

Reward models: `simple_preferred_action`, `shared_base_gaussian`,
`shared_good_bad_heterogeneous`, `consensus_welfare_gaussian`.

The stepsize bases are identical across all experiments and live in
`configspec.DEFAULT_STEPSIZES` rather than on the CLI:
`alpha_pu=0.05`, `beta_status=0.05`, `eta_v=0.1`, `eta_s=0.1`, `eta_J=0.05`, all with
decay 0.01.

## Output files

For an experiment with prefix `<stem>` and mode `<mode>`:

| File | Contents |
| --- | --- |
| `<stem>_runs_<mode>.csv` | one row per (cell, seed), columns exactly `record_columns` |
| `<stem>_aggregate_<mode>.csv` | one row per cell: keys, `n_runs`, then the aggregate spec |
| `<stem>_seed_comparison_<mode>.csv` | a readable subset, sorted for eyeballing seed variation |
| side tables | A: `pu_progression_*`, `pu_agent_traces_*`. E: `actor_rate_agents_*`, `actor_rate_timeseries_*`. D: `*_exit_diagnostics.csv` |
| PNGs | metric line plots and heatmaps, in `--output-dir` |

**Report figures land in a sibling directory, not inside `--output-dir`.** A, B and C
write to `<output-dir>/../final_report_figures/`. That is why the committed layout is
`outputs/exp_a/final_pu_baseline/` next to `outputs/exp_a/final_report_figures/`. Point
`--output-dir` at a named run subdirectory, not at the experiment root, or the report
figures will land one level too high. Experiment E is the exception: it writes to
`<output-dir>/figures/`.

## Reading the results

Three habits, all of which the code is built to support and all of which are easy to
skip:

1. **Every conditional mean has a rate next to it.** `mean_time_to_90pct_followers` with
   `reach_rate`; `mean_recovery_time` with `recovery_rate`; every Experiment E mean with
   `mean_share_at_ceiling` and `driver_share_status`. The aggregate specs deliberately
   order the rates first.
2. **Sentinels are not data.** `-1` means "did not happen" for `time_to_*`,
   `recovery_time`, and `leader_id`. The `where=nonneg` / `positive` filters in the
   aggregate spec drop them; anything you compute yourself downstream must too.
3. **A single-seed cell is an anecdote.** Several committed sweeps use one seed, which is
   fine for checking a mechanism and not fine for a claimed effect size. `ci95_*` columns
   are computed but meaningless at n = 1.

## Testing

```bash
PYTHONPATH=src python3 -m pytest -q                      # everything, ~2.5 min
PYTHONPATH=src python3 -m pytest -q tests/unit_tests     # ~3 s
PYTHONPATH=src python3 -m pytest -q tests/harness_tests  # ~1 s
PYTHONPATH=src python3 -m pytest -q tests/parity_tests   # ~35 s
PYTHONPATH=src python3 -m pytest -q tests/experiment_tests  # ~2 min
```

Four layers, each testing something different:

**`unit_tests/`** — per-module tests of `model/`. Everything is seeded, and an autouse
`poison_global_rng` fixture perturbs `np.random` before every test so that a determinism
test which accidentally depends on the global stream fails loudly instead of passing by
luck.

**`harness_tests/`** — the harness plumbing: axes, aggregation, column ownership,
CSV writing, the plugin protocol.

**`parity_tests/`** — the package against the frozen `benchmark_code.py`, module by
module (distributions, role updates, welfare, Phase 4). Tolerance is
`SOFTMAX_EPS_TOL = 1e-6`: the benchmark divides by `sum(exp) + 1e-8`, so its policies sum
to 0.999999993, worth 5.8e-9 on expected utilities. `test_welfare_parity` pins that
magnitude separately so a real divergence cannot hide inside the tolerance. Section 7 is
order-dependent, so tests comparing role assignment must request the `pinned_shuffle`
fixture. Deleting `benchmark_code.py` retires this suite cleanly — the conftest skips the
directory rather than erroring.

**`experiment_tests/`** — the v2 harnesses against the v1 originals, end to end. These
are **distributional, not exact**, and the conftest explains why: the old engine drew
everything from the global stream, the new one uses `config.runtime.seed` with five
spawned substreams, and no seed pair makes the two consume draws in the same order. A
given seed is therefore a different draw from the same distribution, so comparisons run
over 12-seed ensembles with a Welch t statistic. Exact assertions are still used for
anything downstream of a fixed input: CSV schema, config translation, metric functions.
The active regime (γ=5, `B_R`=0.5, `shared_good_bad_heterogeneous`, 400 steps, N=6) is
chosen so that followers actually form — at the defaults nothing follows anyone, every
role metric is identically zero, and the comparison passes vacuously.
`test_active_regime_actually_forms_followers` guards that.

### Differential check: v2 vs v3

`tools/parity_check.py` is the acceptance criterion for the port. It runs both the legacy
v2 harness and the v3 port with identical arguments into separate temp directories and
compares every CSV cell by cell at `1e-12`.

```bash
PYTHONPATH=src python3 tools/parity_check.py           # quick cases (~2 min)
PYTHONPATH=src python3 tools/parity_check.py --full    # adds async and Experiment D
PYTHONPATH=src python3 tools/parity_check.py -k exp_a  # one case
PYTHONPATH=src python3 tools/parity_check.py --keep    # keep temp output dirs
```

Cases: `exp_a_static`, `exp_a_async`, `exp_b_static`, `exp_c_static`, plus
`exp_b_async`, `exp_d_static`, `exp_d_async` under `--full`. Experiment E has no legacy
counterpart and no case.

This works only because the default `--seed-derivation` is `legacy_global`; see below.

## Reproducibility

Two flags change results and are worth understanding before you compare anything across
runs.

**`--seed-derivation`** — how the engine seed is derived from the run seed.

- `legacy_global` (default): the v2 path. The harnesses called `np.random.seed(s)` and
  never passed a seed to `SystemConfig`, so the compat shim drew the engine seed from the
  freshly-seeded *global* stream via `np.random.randint`. The trajectory is still a
  deterministic function of `s`, just through an extra layer of laundering. Required for
  byte-identical parity with the committed CSVs.
- `direct`: `runtime.seed = s`. What the engine's API intends, what the unit tests
  assume, and what anyone reading the code would expect. Produces different — equally
  valid — trajectories.

**`--schedule-rng`** — the source of randomness for asynchronous role-update timers.

- `global` (default): `np.random`, seeded per run. Bit-compatible with existing outputs.
- `stream`: a dedicated `Generator` spawned from the run seed. Process- and thread-safe,
  which matters if you parallelise, but it changes async results.

The corollary: **do not mix outputs produced under different settings of either flag in
one figure.** Start new work on `direct` + `stream` if you do not need parity with the
committed CSVs; keep `legacy_global` + `global` if you do.

## Lineage

| Layer | What it is | Status |
| --- | --- | --- |
| `src/benchmark_code.py` | upstream `src/code_debugged.py`, verbatim except that `self.results[k].append(...)` became `self.results.setdefault(k, []).append(...)` throughout. Behaviour-identical; the change only stops a missing key from raising. | frozen reference |
| `src/experiments/v1/` | upstream `experiments/*.py`, importing `benchmark_code`. `reputation_status_scaling.py` has no upstream counterpart — upstream's `reputation_scaling.py` fixed κ = 0, and this one sweeps both γ and κ. | frozen reference |
| `src/experiments/v2/` | the same harnesses with one import line changed to `experiments.v2.compat`, plus two calls renamed where the engine API differs (`_compute_true_reputation_vector` → `_true_reputation`). The diff is ~10 lines per file, deliberately, so the experiment-level tests measure the **engine** change rather than a hand-translation of 8,000 lines. | frozen reference |
| `src/experiments/v2/compat.py` | adapter presenting the old flat-kwargs `SystemConfig` / `MultiAgentSystem` API on top of `model/`. Explicitly temporary — everything it does is a translation, and each translation is a place a future reader could be misled about what the engine actually offers. | to be deleted |
| `src/experiments/v3/` | the ports: A/B/C/D declared over the shared harness with byte-identical CSV output to v2, plus E, which is new. 987 → ~275 lines for A; 1,684 → ~172 for D. | **current** |

The v1 and v2 trees exist to be *run*, not read. They are the oracle that
`tools/parity_check.py` and `tests/experiment_tests/` compare against. Once the port is
settled they can go, and with them `compat.py` and `benchmark_code.py` — the parity
conftest is already written to skip cleanly when `benchmark_code.py` disappears.

## Known issues

Ordered roughly by how likely each is to cost you a result.

1. **Experiment E's defaults dict uses hyphenated keys.** `exp_e_actor_rate_separation.py`
   passes `"num-states": 7`, `"num-actions": 3`, `"reward-clip-min"`, `"reward-clip-max"`
   into `add_core_arguments(defaults=...)`, which looks them up by *dest* name
   (`num_actions`, …). The hyphenated keys are silently ignored, so the effective
   defaults are `num_states=3` and `num_actions=2`, not 7 and 3. (`num_states` would not
   take effect even spelled correctly — `add_core_arguments` does not declare it; E
   declares it itself with `default=3`.) The reward-clip values happen to match the
   harness defaults, so those two are harmless. **Pass `--num-states` and `--num-actions`
   explicitly** until this is fixed, as the actual command above does.

2. **Two unit tests fail on `main`.** `tests/unit_tests/test_instrumentation.py::test_full_recorder_exposes_the_same_wants_flags`
   and `::test_dense_history_implies_compact_histories` reference
   `FullRecorder.wants_compact_histories`; the attribute is
   `wants_compact_history` (singular). The tests are wrong, not the engine. 752 of 754
   tests pass.

3. **`pytest` markers are declared in a text file, not in config.**
   `tests/experiment_tests/pytest_markers.txt` contains the `[tool.pytest.ini_options]`
   block that should be in `pyproject.toml`. Until it is moved, a bare `pytest` runs the
   slow suites too (~2.5 min instead of ~4 s) and emits `PytestUnknownMarkWarning`. Also
   note that `parity_tests/` carries no `pytestmark`, so `-m parity` would select nothing
   even once the markers are registered — select it by path instead.

4. **`requires-python = ">=3.8"` is wrong.** `argparse.BooleanOptionalAction` (3.9+) is
   used in `harness/cli.py`, `extras/perturbation.py`, and two v3 experiments, and nine
   `model/` modules use PEP 585 generics (`tuple[...]`, `dict[...]`) in annotations
   without `from __future__ import annotations`. The real floor is **3.9**.

5. **`leader_actor_rates` bloats Experiment B's runs CSV.** `ActorRateTrackerPlugin`
   serialises the leader's full per-step rate series into a single CSV cell, including
   the `np.float64(...)` reprs. The committed 6-run file is 574 KB, nearly all of it that
   one column, and it is not machine-readable without `eval`. It should be a side table
   like Experiment E's `actor_rate_timeseries`, or at minimum downsampled and formatted
   as plain floats.

6. **`--mode` is required for Experiment D despite `parser.set_defaults(mode="static")`.**
   `set_defaults` does not clear `required=True` on the underlying argument, so the
   default is unreachable and the flag must be passed. Harmless, but the code reads as
   though `--mode` were optional there.

7. **`docs/HARNESS.md` does not exist.** `harness/cli.py` points at it from the
   `--seed-derivation` and `--schedule-rng` help text. The
   [Reproducibility](#reproducibility) section above covers what it would have said.

8. **The engine prints during `step()`.** `run_single` wraps the loop in
   `redirect_stdout` to suppress it. That is a bug workaround, not a feature: the fix is
   for the engine to route progress through the `Recorder` / `progress_printer` instead
   of stdout, at which point the guard can go.

9. **Experiment D's async path does not refresh tracked state after a role update**,
   unlike A/B/C, so step *t* does not reflect the post-update follower graph and D's
   follower series lags the others by a step. Reproduced deliberately via
   `async_refresh=False` for parity; almost certainly an oversight in the original rather
   than a choice.

10. **`.DS_Store` files are committed** at the repo root and in several subdirectories.
    Add `.DS_Store` to `.gitignore` and `git rm --cached` them.

## License

MIT. See `LICENSE`.
