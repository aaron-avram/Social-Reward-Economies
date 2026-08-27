"""
Ensemble comparison of the two engines under the parameters the harnesses use.

This is the test that would catch a porting error in compat.py: a config field
routed to the wrong group, a stepsize decay dropped, a mode string mistranslated.
Any of those shifts the outcome distribution well outside sampling error.
"""
import numpy as np
import pytest

from .conftest import ACTIVE, N_SEEDS, run_ensemble, welch_t

METRICS = ["max_followers", "n_reputation", "n_status", "n_personal_utility",
           "has_leader", "final_welfare", "mean_welfare", "mean_actor_rate",
           "n_role_updates"]


@pytest.fixture(scope="module")
def ensembles(old_engine, new_engine):
    return run_ensemble(old_engine), run_ensemble(new_engine)


def test_active_regime_actually_forms_followers(ensembles):
    """
    Guard against a vacuous comparison. If nothing follows, every role metric is
    zero on both sides and the whole suite passes without testing anything.
    """
    old, new = ensembles
    for label, runs in (("old", old), ("new", new)):
        rate = np.mean([r["has_leader"] for r in runs])
        assert rate > 0.5, f"{label}: followers formed in only {rate:.0%} of runs"


@pytest.mark.parametrize("metric", METRICS)
def test_metric_distributions_agree(ensembles, metric):
    old, new = ensembles
    a = [r[metric] for r in old]
    b = [r[metric] for r in new]
    t = welch_t(a, b)
    assert t < 4.0, (f"{metric}: old {np.mean(a):.4f} vs new {np.mean(b):.4f} "
                     f"(t={t:.2f}, n={N_SEEDS})")


@pytest.mark.parametrize("gamma", [0.0, 2.0, 5.0, 10.0])
def test_gamma_response_agrees(old_engine, new_engine, gamma):
    """
    gamma drives follower formation, so this checks the two engines respond the
    same way across the sweep axis the experiments vary — not just at one point.
    """
    old = run_ensemble(old_engine, n_seeds=8, gamma=gamma)
    new = run_ensemble(new_engine, n_seeds=8, gamma=gamma)
    for metric in ("max_followers", "n_reputation", "mean_welfare"):
        t = welch_t([r[metric] for r in old], [r[metric] for r in new])
        assert t < 4.0, f"gamma={gamma}, {metric}: t={t:.2f}"


@pytest.mark.parametrize("kappa", [0.0, 2.0, 10.0])
def test_kappa_response_agrees(old_engine, new_engine, kappa):
    """kappa gates STATUS entry — the other sweep axis."""
    old = run_ensemble(old_engine, n_seeds=8, kappa=kappa)
    new = run_ensemble(new_engine, n_seeds=8, kappa=kappa)
    for metric in ("n_status", "max_followers", "mean_welfare"):
        t = welch_t([r[metric] for r in old], [r[metric] for r in new])
        assert t < 4.0, f"kappa={kappa}, {metric}: t={t:.2f}"


@pytest.mark.parametrize("reward_model", [
    "simple_preferred_action", "shared_base_gaussian",
    "shared_good_bad_heterogeneous", "consensus_welfare_gaussian",
])
def test_every_reward_model_agrees(old_engine, new_engine, reward_model):
    old = run_ensemble(old_engine, n_seeds=8, reward_model=reward_model)
    new = run_ensemble(new_engine, n_seeds=8, reward_model=reward_model)
    for metric in ("mean_welfare", "max_followers"):
        t = welch_t([r[metric] for r in old], [r[metric] for r in new])
        assert t < 4.0, f"{reward_model}, {metric}: t={t:.2f}"


def test_role_update_schedule_matches_exactly(old_engine, new_engine):
    """
    The epoch schedule consumes no randomness, so this one IS exact: both engines
    must fire role updates at identical timesteps.
    """
    old = run_ensemble(old_engine, n_seeds=1)
    new = run_ensemble(new_engine, n_seeds=1)
    assert old[0]["n_role_updates"] == new[0]["n_role_updates"]
