"""
Unit tests for the harness layer.

These cover the parts that were previously unreachable without running a full
sweep: the metric functions, the grid enumeration, the aggregation spec, and the
column-ownership check. Each is a pure function of arrays or dicts, so the tests
run in milliseconds and pin down edge cases (empty series, no leader, single
seed) that a sweep would never exercise.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from experiments.harness import metrics
from experiments.harness.aggregate import Derived, Triple, aggregate, finite, nonneg
from experiments.harness.axes import Axis, Grid, cell_tag, format_num
from experiments.harness.cli import (
    interval_seq_from_epochs,
    parse_csv_floats,
    parse_csv_ints,
    parse_role_update_epochs,
    parse_role_update_T_seq,
    resolve_seeds,
)
from experiments.harness.plugins import MetricsPlugin, RunPlugin, check_column_ownership


# ------------------------------------------------------------------ axes ---

def test_grid_orders_cells_outer_seeds_inner():
    grid = Grid(
        axes=(Axis.of("a", ["x", "y"]), Axis.of("b", [1, 2])),
        seeds=(0, 1),
    )
    runs = list(grid.runs())
    assert len(runs) == 8 == len(grid)
    # First axis varies slowest, seed fastest -- this order is what makes the
    # ported CSVs match the legacy nested loops row for row.
    assert [r[0]["a"] for r in runs] == ["x"] * 4 + ["y"] * 4
    assert [r[1] for r in runs] == [0, 1] * 4


def test_zero_axis_grid_is_one_cell():
    grid = Grid(axes=(), seeds=(0, 1, 2))
    assert [c for c, _ in grid.runs()] == [{}, {}, {}]
    assert len(grid) == 3


def test_grid_rejects_duplicate_axis_names():
    with pytest.raises(ValueError, match="duplicate axis"):
        Grid(axes=(Axis.of("a", [1]), Axis.of("a", [2])), seeds=(0,))


def test_axis_rejects_empty_values():
    with pytest.raises(ValueError, match="no values"):
        Axis.of("a", [])


@pytest.mark.parametrize(
    "value,expected",
    [(5.0, "5"), (2, "2"), (0.5, "0p5"), (-1.5, "m1p5")],
)
def test_format_num_is_filename_safe(value, expected):
    assert format_num(value) == expected


def test_cell_tag_combines_axes():
    assert cell_tag({"gamma": 5.0, "kappa": 2.0}) == "g5_k2"


# --------------------------------------------------------------- parsers ---

def test_parse_csv_helpers_ignore_blanks():
    assert parse_csv_ints(" 1, 2 ,,3 ") == [1, 2, 3]
    assert parse_csv_floats("0,0.5") == [0.0, 0.5]
    assert parse_csv_ints("") == []


def test_role_update_epochs_dedupe_sort_and_drop_nonpositive():
    assert parse_role_update_epochs("300,100,100,0,-5") == [100, 300]


def test_role_update_T_seq_preserves_order_drops_nonpositive():
    assert parse_role_update_T_seq("2000,0,3000") == [2000, 3000]


def test_interval_seq_from_epochs_is_first_differences_after_s0():
    assert interval_seq_from_epochs(s0=100, epochs=[300, 600, 1000]) == [200, 300, 400]


def test_interval_seq_skips_epochs_at_or_before_s0():
    assert interval_seq_from_epochs(s0=500, epochs=[100, 300, 700]) == [200]


def test_selected_seeds_override_range():
    args = argparse.Namespace(selected_seeds="5,3,3", seeds=10, seed_start=0)
    assert resolve_seeds(args) == [3, 5]
    args = argparse.Namespace(selected_seeds="", seeds=3, seed_start=7)
    assert resolve_seeds(args) == [7, 8, 9]


# --------------------------------------------------------------- metrics ---

def test_leader_series_breaks_ties_by_lowest_id_and_marks_leaderless():
    counts = np.array([[0, 0, 0], [2, 2, 1], [1, 3, 0]], dtype=float)
    assert metrics.leader_series_from_follower_counts(counts).tolist() == [-1, 0, 1]


def test_hysteretic_series_holds_incumbent_within_margin():
    # Agent 0 leads first; agent 1 pulls ahead by exactly 1, which is NOT a
    # strict excess of the margin, so the incumbent is retained.
    counts = np.array([[3, 0], [3, 4], [3, 6]], dtype=float)
    assert metrics.leader_series_hysteretic(counts, margin=1).tolist() == [0, 0, 1]


def test_hysteretic_margin_zero_matches_plain_series():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=(40, 4)).astype(float)
    assert (
        metrics.leader_series_hysteretic(counts, margin=0).tolist()
        == metrics.leader_series_from_follower_counts(counts).tolist()
    )


def test_leader_switches_ignores_leaderless_steps():
    # -1 entries are dropped, so 0 -> -1 -> 0 is not two switches.
    assert metrics.leader_switches(np.array([0, -1, 0, 1])) == 1
    assert metrics.leader_switches(np.array([-1, -1])) == 0
    assert metrics.leader_switches(np.array([], dtype=int)) == 0


def test_time_to_threshold_is_one_indexed_and_signals_never():
    assert metrics.time_to_threshold(np.array([0, 1, 5, 2]), 5) == 3
    assert metrics.time_to_threshold(np.array([0, 1]), 5) == -1


def test_tail_share_variants_agree_when_denominator_positive():
    counts = np.array([[1, 4], [2, 6]], dtype=float)
    a = metrics.tail_top_follower_share(counts, tail_window=2, denom=8)
    b = metrics.tail_top_follower_share_elementwise(counts, tail_window=2, denom=8)
    assert a == pytest.approx(b)
    assert a == pytest.approx(np.mean([4, 6]) / 8)


def test_tail_share_guards_empty_and_nonpositive_denominator():
    empty = np.array([], dtype=float).reshape(0, 0)
    assert metrics.tail_top_follower_share(empty, 5, 4) == 0.0
    assert metrics.tail_top_follower_share(np.ones((2, 2)), 5, 0) == 0.0


def test_tail_status_leader_share_divides_by_steps_with_a_leader():
    roles = np.array([["status", "reputation"], ["status", "reputation"]], dtype=object)
    leaders = np.array([-1, 0])
    # Only one of the two tail steps has a leader, and on that step the leader
    # is in STATUS, so the share is 1.0 -- not 0.5.
    assert metrics.tail_status_leader_share(roles, leaders, 2) == pytest.approx(1.0)


def test_tail_status_agent_share_is_over_all_agent_steps():
    roles = np.array([["status", "reputation"], ["status", "status"]], dtype=object)
    assert metrics.tail_status_agent_share(roles, 2) == pytest.approx(0.75)


def test_mean_std_ci_single_value_has_zero_spread():
    mean, std, ci = metrics.mean_std_ci([3.0])
    assert (mean, std, ci) == (3.0, 0.0, 0.0)


def test_mean_std_ci_empty_is_all_nan():
    assert all(np.isnan(v) for v in metrics.mean_std_ci([]))


def test_mean_std_ci_uses_sample_std_and_normal_interval():
    mean, std, ci = metrics.mean_std_ci([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)  # ddof=1
    assert ci == pytest.approx(1.96 / np.sqrt(3))


def test_role_to_label_handles_enum_and_string():
    class Fake:
        value = "STATUS"

    assert metrics.role_to_label(Fake()) == "status"
    assert metrics.role_to_label("Reputation") == "reputation"


# ------------------------------------------------------------- aggregate ---

def _records():
    return [
        {"mode": "static", "g": 0, "t": 10, "x": 1.0},
        {"mode": "static", "g": 0, "t": -1, "x": 2.0},
        {"mode": "static", "g": 1, "t": 4, "x": float("nan")},
    ]


def test_aggregate_groups_and_counts():
    rows = aggregate(_records(), group_by=["mode", "g"], spec=[Triple("x")])
    assert [r["g"] for r in rows] == [0, 1]
    assert [r["n_runs"] for r in rows] == [2, 1]
    assert rows[0]["mean_x"] == pytest.approx(1.5)


def test_triple_where_filters_sentinels():
    rows = aggregate(_records(), group_by=["mode"], spec=[Triple("t", where=nonneg)])
    # -1 means "never reached" and must not be averaged in as a duration.
    assert rows[0]["mean_t"] == pytest.approx((10 + 4) / 2)


def test_triple_fallback_used_when_filter_empties_the_group():
    recs = [{"mode": "s", "t": -1}, {"mode": "s", "t": -1}]
    rows = aggregate(recs, group_by=["mode"],
                     spec=[Triple("t", where=nonneg, fallback=[-1.0])])
    assert rows[0]["mean_t"] == -1.0


def test_triple_finite_only_drops_nan():
    rows = aggregate(_records(), group_by=["mode"], spec=[Triple("x", finite_only=True)])
    assert rows[0]["mean_x"] == pytest.approx(1.5)


def test_triple_without_finite_only_propagates_nan():
    rows = aggregate(_records(), group_by=["mode"], spec=[Triple("x")])
    assert np.isnan(rows[0]["mean_x"])


def test_derived_column_sees_the_whole_group():
    spec = [Derived("reach_rate", lambda g: sum(r["t"] >= 0 for r in g) / len(g))]
    rows = aggregate(_records(), group_by=["mode"], spec=spec)
    assert rows[0]["reach_rate"] == pytest.approx(2 / 3)


def test_triple_stem_overrides_column_names():
    assert Triple("x", stem="y").columns == ("mean_y", "std_y", "ci95_y")


# --------------------------------------------------------------- plugins ---

class _P(RunPlugin):
    def __init__(self, name, columns):
        self.name = name
        self.columns = tuple(columns)


def test_column_ownership_accepts_exact_cover():
    check_column_ownership(
        [_P("a", ("x",)), _P("b", ("y",))],
        key_columns=("mode", "seed"),
        record_columns=("mode", "seed", "x", "y"),
    )


def test_column_ownership_rejects_duplicate_claims():
    with pytest.raises(ValueError, match="declared by both"):
        check_column_ownership(
            [_P("a", ("x",)), _P("b", ("x",))],
            key_columns=(),
            record_columns=("x",),
        )


def test_column_ownership_rejects_unclaimed_schema_column():
    with pytest.raises(ValueError, match="unclaimed"):
        check_column_ownership(
            [_P("a", ("x",))], key_columns=(), record_columns=("x", "z")
        )


def test_column_ownership_rejects_plugin_column_missing_from_schema():
    with pytest.raises(ValueError, match="not in schema"):
        check_column_ownership(
            [_P("a", ("x", "q"))], key_columns=(), record_columns=("x",)
        )


def test_metrics_plugin_enforces_its_declared_contract():
    plugin = MetricsPlugin("m", ("a", "b"), lambda ctx: {"a": 1})
    with pytest.raises(RuntimeError, match="missing=\\['b'\\]"):
        plugin.measure(None)

    plugin = MetricsPlugin("m", ("a",), lambda ctx: {"a": 1, "extra": 2})
    with pytest.raises(RuntimeError, match="unexpected=\\['extra'\\]"):
        plugin.measure(None)
