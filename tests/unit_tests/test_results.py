"""results.py — the typed schema that replaced the 30-key dict."""
import numpy as np
import pytest

from model.results import SimulationResults, StepRecord


def rec(t=1, **kw):
    base = dict(
        t=t, follower_counts=[0, 1], actor_count=2, participant_count=2,
        online_active_actor_payoff_sum=1.0, paper_welfare_all_agents=3.0,
        paper_welfare_followers_only=2.0, status_count=0, pu_count=2, rep_count=0,
        role_label=["personal_utility"] * 2,
    )
    base.update(kw)
    return StepRecord(**base)


def test_append_grows_core_histories():
    r = SimulationResults()
    for t in (1, 2, 3):
        r.append(rec(t))
    assert r.step_count() == 3 and len(r) == 3
    assert len(r.actor_counts) == 3


def test_social_welfare_aliases_followers_only():
    r = SimulationResults()
    r.append(rec(paper_welfare_followers_only=7.5))
    assert r.social_welfare == [7.5]


def test_role_update_times_only_on_update_steps():
    r = SimulationResults()
    for t in (1, 2, 3):
        r.append(rec(t), role_updated=(t == 2))
    assert r.role_update_times == [2]


def test_optional_fields_are_skipped_when_none():
    r = SimulationResults()
    r.append(rec())
    assert r.norm_consensus == [] and r.expected_utilities == []


def test_optional_fields_land_when_present():
    r = SimulationResults()
    r.append(rec(norm_consensus=0.5, expected_utilities={0: 1.0},
                 dense_reputation=np.zeros((2, 2))))
    assert r.norm_consensus == [0.5]
    assert len(r.dense_reputation_history) == 1


def test_overwrite_last_updates_the_refreshed_subset():
    r = SimulationResults()
    r.append(rec(1)); r.append(rec(2))
    r.overwrite_last(rec(2, follower_counts=[9, 9], paper_welfare_followers_only=8.0))
    assert r.follower_counts[-1] == [9, 9]
    assert r.social_welfare[-1] == 8.0


def test_overwrite_last_leaves_the_unrefreshed_fields_alone():
    """refresh_last_tracked_state deliberately skips activation counts (1320-1404)."""
    r = SimulationResults()
    r.append(rec(1, actor_count=5))
    r.overwrite_last(rec(1, actor_count=99, follower_counts=[7, 7]))
    assert r.actor_counts == [5]


def test_overwrite_last_on_empty_history_is_a_noop():
    SimulationResults().overwrite_last(rec())


def test_step_count_works_under_light_tracking():
    """norm_consensus is FULL-only, so len(norm_consensus) is not the step count."""
    r = SimulationResults()
    for t in range(5):
        r.append(rec(t))
    assert r.step_count() == 5


def test_validate_accepts_a_consistent_run():
    r = SimulationResults()
    for t in range(4):
        r.append(rec(t, norm_consensus=float(t)))
    r.validate()


def test_validate_catches_a_ragged_history():
    r = SimulationResults()
    r.append(rec(1, norm_consensus=0.1))
    r.append(rec(2))
    with pytest.raises(AssertionError):
        r.validate()


def test_previously_undeclared_fields_exist():
    """These were created by setdefault in the original and absent from the
    declared dict; a KeyError would only show up mid-sweep."""
    r = SimulationResults()
    for name in ("social_welfare", "status_counts", "pu_counts", "rep_counts",
                 "online_active_actor_payoff_sum", "role_update_diagnostics",
                 "true_reputation_checkpoints", "estimate_consensus_checkpoints",
                 "rate_audit_checkpoints"):
        assert hasattr(r, name), name


def test_summary_fields_default_to_unset():
    r = SimulationResults()
    assert r.final_roles is None and r.final_followers is None
    assert r.opinion_leader == -1


def test_npz_roundtrip_preserves_core_history(tmp_path):
    r = SimulationResults()
    for t in range(3):
        r.append(rec(t))
    path = str(tmp_path / "out.npz")
    r.to_npz(path, {"schema_version": 1})
    back = SimulationResults.from_npz(path)
    assert back.step_count() == 3
    assert back.paper_welfare_all_agents == [3.0, 3.0, 3.0]


def test_from_npz_rejects_a_stale_schema(tmp_path, monkeypatch):
    import model.results as RES
    r = SimulationResults(); r.append(rec())
    path = str(tmp_path / "old.npz")
    r.to_npz(path, {})
    monkeypatch.setattr(RES, "SCHEMA_VERSION", 999)
    with pytest.raises(ValueError):
        SimulationResults.from_npz(path)
