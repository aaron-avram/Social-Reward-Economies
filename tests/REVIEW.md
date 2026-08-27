# Code review — `socialreward` package

Assembled the twelve modules into a package and ran the suite in `tests/`:
**143 passed, 107 failed.** The failures collapse to **17 distinct root causes**,
listed below in fix order. Fixing the first four clears ~85 of the 107.

---

## Blockers — nothing runs until these are fixed

### 1. `Generator.randn` does not exist — 49 failures
`agent.py:76-77`

```python
self.state.weights_pu = self.rng.randn(self.dims.num_states, self.dims.num_actions) * 0.1
```

`randn` is the legacy `RandomState` name. On a `Generator` it is
`standard_normal(size=...)`. Every test that constructs an `Agent` dies here.

```python
self.state.weights_pu = self.rng.standard_normal(
    (self.dims.num_states, self.dims.num_actions)) * 0.1
```

Also: `self.rng` is stored on the agent. After construction the agent draws
nothing — dropping the attribute makes that guarantee structural.

### 2. `SystemConfig.dimensions` vs `config.dims` — 29 failures
`config.py:210` declares `dimensions`; `system.py` and `plots.py` read `.dims`
in eleven places. Rename the field to `dims`.

### 3. `rng.init.*` inside reward builders — 4 failures
`rewards.py:125,131,158,171,175,178,187`

The caller already strips the bundle and passes a bare `Generator`, so
`rng.init.normal(...)` is an `AttributeError`. `SharedBaseGaussian` gets this
right; the other two do not. Also `rng.init.randint` is doubly wrong —
`Generator` has `integers`, not `randint`.

### 4. `params.kind` vs `params.model`
`rewards.py:199` indexes the registry with `params.kind`; `config.py:143`
declares `model`. One-line fix either way — but pick `kind`, since
`params.model` inside a module full of models reads badly.

### 5. `_actors_act` is `NotImplementedError`, and its body is in `_role_based_updates`
`system.py:111` raises; `system.py:124-149` contains the Phase-2 body under the
Phase-3 name. So **Phase 3 does not exist at all** — no policy-gradient updates,
no `J^pu`/`J^s` estimate updates, no reputation-reward estimate. The simulation
would run and produce numbers with no learning in it.

Move the body back to `_actors_act` and write `_role_based_updates` from
lines 1961-2005 of the original.

Inside that misplaced body, two more:
- `payoffs = dict[int, float] = {}` (line 133) — chained assignment binds the
  *type object* to `payoffs`, then `{}`. It happens to work; it is not what you
  meant.
- `agent.select_action(s, float(uniforms[k]), leader_weights)` passes the whole
  dict where an `(S, A)` array is expected. Needs `leader_weights.get(k)`.

---

## Logic bugs — these run, and give wrong answers

### 6. `_redirect_target` checks the wrong thing
`roleupdate.py:137-138`

```python
target_was_follower = state.followers[best_k] is not None
return (state.followers[best_k] if target_was_follower else best_k, ...)
```

`state.followers[best_k]` is the **set of agents following best_k**. A set is
never `None`, so the branch always fires and the function returns *a set* as the
new target — which then raises `TypeError: unhashable type: set` at
`state.followers[best_k].add(i)`.

[ROLE-3] asks the opposite question: is `best_k` itself *following* someone?

```python
def _redirect_target(best_k, state):
    leader = state.following[best_k]
    if leader is None:
        return best_k, False
    return leader, True
```

### 7. `collect_signals` indexes column −1 for an unresolved leader
`roleupdate.py:345-346`

```python
target_rep=(0.0 if rep_L[i] is None else float(rep_s[i, rep_L[i]]))
```

`rep_L` is an int array; `NO_LEADER` is `-1`, never `None`. So an agent with no
leader reads `s[i, -1]` — the **last agent's** reputation — and may follow on
the strength of it. Compare against `NO_LEADER`.

### 8. `step2_status` runs its body twice
`roleupdate.py:261-284`. The first block (261-271) is a leftover; the second
(273-284) is the guard-clause version. Harmless only because the operation is
idempotent, and the first block skips `rec.role_update_decision`. Delete 261-271.

### 9. `resolve_root_leader` crashes on an unfollowed agent
`welfare.py:174-176`

```python
current = int(following[agent_id])          # TypeError when None
if current is None:                          # unreachable
    return int(agent_id) if follower_counts > 0 else -1   # list > int
```

Three bugs in three lines: the `int()` precedes the `None` check, the check is
therefore dead, and `follower_counts` is a sequence being compared to an int.

```python
    leader = following[agent_id]
    if leader is None:
        return int(agent_id) if follower_counts[agent_id] > 0 else -1
```

### 10. `np.mx` typo
`welfare.py:141` — `np.max`. Kills every `true_reputation` call.

### 11. `RewardParams.__post__init__` never runs
`config.py:161` — double underscore in the middle. Python calls
`__post_init__`. The `order_gap` validation is silently dead.

### 12. Two distinct `AgentRole` enums
`agent.py:15` and `roles.py:7` define the same enum twice. Members of different
Enum classes are never equal, so a `role is AgentRole.STATUS` check comparing
across the boundary silently returns `False`. Delete one — keep `roles.py` and
have `agent.py` import from it, or drop `roles.py` entirely.

### 13. `AgentSignals` is missing `rep_row` — 5 failures
`instrumentation.py:139-149` reads `sig.rep_row[...]` for the currently-followed
agent and the pre-perturbation leader. `AgentSignals` has five fields, none of
them `rep_row`. Add `rep_row: np.ndarray` and fill it in `collect_signals`.

### 14. `wants_compact_histories` is missing — 3 failures
`system.py:339` reads `self.rec.wants_compact_histories`; neither
`NullRecorder` nor `FullRecorder` defines it. Add to both, and make
`FullRecorder` return `True` whenever `dense_history` is on (the original's
`enable_small_n_trace_export` set both flags at 808-809).

### 15. `role_update_begin` called with the wrong arity
`roleupdate.py:392` — `rec.role_update_begin(state, signals, updatable, params)`;
the protocol is `(t, state, signals, updatable, params)`. `t` never arrives, so
every audit row's `"t"` is wrong.

Same file, line 186: `rec.role_update_step1(i, _effective_threshold=B_i, ...)` —
the seeded row key is `effective_threshold`, so this writes a *new* key and
leaves the real one at `None`.

### 16. Param groups alias across `replace()`
`test_config.py::test_param_groups_are_frozen_or_replace_is_deep` fails.
`dataclasses.replace(cfg, ...)` copies the top level only, so
`derived.algorithm is base.algorithm`. A sweep doing
`cfg.algorithm.gamma = g` mutates every config in the process. Freeze the five
param groups (`@dataclass(frozen=True)`) so the mutation raises instead.

### 17. Skeleton residue
`test_package.py` flags `TODO` markers still in `reputation.py`, `roleupdate.py`,
`system.py`, `welfare.py`, and one `raise NotImplementedError` in `system.py`.
Worth clearing before the baseline commits — a stale TODO in a file you will diff
eight times is noise.

---

## Smaller notes, not test failures

- `reputation.update_personal_benefits` is **dead code**: `phase4` inlines the
  same six lines rather than calling it. Either call it or delete it; two copies
  of Eq. (4) is exactly the drift the refactor was meant to end.
- `eq9_averaging_ids` and `leader_update_ids` return **lists**, but `phase4`
  annotates them as `np.ndarray` and the `Phase4Trace` comprehension assumes
  iterability. Works today; return `np.array(..., dtype=int)` for consistency
  with `gossip_targets`.
- `RoleUpdateState.validate` checks the partition but not the follow-graph
  invariants its own docstring promises (`j in followers[i] <=> following[j]==i`,
  no self-follow, no chains). The system-level tests check these instead; worth
  moving into `validate`.
- `system._adopt_leader_behavior` does `leader_weights[i]` — `KeyError` for a
  REPUTATION agent with `following is None`. Use `.get(i)` and let
  `adopt_behavior`'s guard handle it.
- `welfare.current_policies` has the same exposure via
  `leader_weights[agent.agent_id]`.
- `system._maybe_update_roles` has no `update_roles()` public wrapper, so the
  three async harnesses that called `_update_roles_sequential(candidates)` have
  no entry point.

---

## Test suite

`tests/`, 250 tests across 12 files. Run with `pytest` from the package root.

| file | tests | covers |
|---|---|---|
| `test_config.py` | 15 | dataclass wiring, validation, enum round-trip, sweep aliasing |
| `test_rng.py` | 5 | stream independence, the global-RNG tripwire |
| `test_rewards.py` | 16 | all four models, ABC contract, registry completeness |
| `test_agent.py` | 20 | leaf-ness, inverse-CDF, hysteresis of the rate driver |
| `test_reputation.py` | 40 | Eq. (4), Eq. (9), leader selection, all 6 mode combos |
| `test_roleupdate.py` | 40 | the three steps, graph invariants, epoch schedule |
| `test_welfare.py` | 17 | einsum orientation, welfare decomposition, chain walking |
| `test_results.py` | 15 | append/overwrite asymmetry, schema, npz round-trip |
| `test_instrumentation.py` | 15 | first-write-wins, JSON-serialisability of rows |
| `test_system.py` | 25 | determinism, CRN, graph consistency, both tracking modes |
| `test_plots.py` | 8 | renders under FULL and LIGHT, unfinalised runs |
| `test_package.py` | 30 | import hygiene, no matplotlib below plots, acyclic graph |

Three tests deserve special mention because they encode decisions from the
refactor that are otherwise invisible:

- `test_resolve_missing_leaders_is_idempotent_and_draws_once_per_agent` asserts
  the **draw count**, not just the result. An unconditional recompute would pass
  a value check and still desynchronise the tiebreak stream from step one.
- `test_decision_is_first_write_wins` is the one test for original line 2368 —
  the only conditional decision write in Section 7.
- `test_expected_observer_utilities_orientation` catches a transposed `einsum`,
  which returns a plausible `(N, N)` array either way.

Not yet covered, and worth adding once the code runs:
- **Parity against `code_debugged.py`** — the highest-value test in the whole
  effort. Run both engines from an identical seed and assert `allclose` on
  `v`, `s`, `L`, roles, and follower sets at t=200 and t=2000.
- A large-N smoke test (N=100, T=500) to catch anything that only shows up at
  scale.
