"""
End-to-end distributional parity.

Bit-exact end-to-end comparison is impossible: the benchmark draws everything from
one global stream (weights, actions, tie-breaks, shuffle), the package uses five
spawned substreams and batches per-agent draws. No seed pair aligns them.

What IS comparable is the distribution of outcomes over many seeds. A wrong
equation, a dropped update, or a sign error moves the mean well outside the
sampling error; a difference in draw order does not.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from .harness import load_benchmark
from model.config import (
    Dimensions, RuntimeParams, ScheduleParams, SystemConfig, TrackingMode,
)
from model.system import MultiAgentSystem

bm = load_benchmark()
N, T, REPS = 6, 400, 60


def run_benchmark(seed):
    np.random.seed(seed)
    s = bm.MultiAgentSystem(bm.SystemConfig(
        num_agents=N, num_time_steps=T, role_update_base_interval=50))
    for _ in range(T):
        s.step()
    return {
        "welfare_all": s.results["paper_welfare_all_agents"][-1],
        "welfare_followers": s.results["paper_welfare_followers_only"][-1],
        "max_followers": max(len(a.state.followers) for a in s.agents),
        "n_reputation": sum(1 for a in s.agents if a.state.role.value == "reputation"),
        "n_status": sum(1 for a in s.agents if a.state.role.value == "status"),
        "mean_actor_rate": float(np.mean([a.state.actor_interaction_rate for a in s.agents])),
    }


def run_package(seed):
    cfg = SystemConfig(
        dims=Dimensions(num_agents=N, num_states=3, num_actions=2),
        runtime=RuntimeParams(seed=seed, num_time_steps=T,
                              tracking_mode=TrackingMode.FULL),
        schedule=ScheduleParams(role_update_base_interval=50),
    )
    s = MultiAgentSystem(cfg)
    res = s.simulate()
    return {
        "welfare_all": res.paper_welfare_all_agents[-1],
        "welfare_followers": res.paper_welfare_followers_only[-1],
        "max_followers": max(len(a.state.followers) for a in s.agents),
        "n_reputation": sum(1 for a in s.agents if a.state.role.value == "reputation"),
        "n_status": sum(1 for a in s.agents if a.state.role.value == "status"),
        "mean_actor_rate": float(np.mean([a.state.actor_interaction_rate for a in s.agents])),
    }


@pytest.fixture(scope="module")
def samples():
    b = [run_benchmark(s) for s in range(REPS)]
    p = [run_package(s) for s in range(REPS)]
    return b, p


METRICS = ["welfare_all", "welfare_followers", "max_followers",
           "n_reputation", "n_status", "mean_actor_rate"]


@pytest.mark.parametrize("metric", METRICS)
def test_means_agree_within_sampling_error(samples, metric):
    """Welch's t-test on the difference of means. A systematic algorithmic
    difference shows up as |t| well above 3; draw-order noise does not."""
    b, p = samples
    xb = np.array([r[metric] for r in b], dtype=float)
    xp = np.array([r[metric] for r in p], dtype=float)
    diff = abs(xb.mean() - xp.mean())
    # Floor: welfare is near-deterministic here (variance ~1e-18), so the softmax
    # epsilon alone would produce an astronomical t. Anything under 1e-6 is that
    # known difference, not an algorithmic one.
    if diff < 1e-6:
        return
    se = np.sqrt(xb.var(ddof=1) / len(xb) + xp.var(ddof=1) / len(xp))
    assert se > 0, f"{metric}: degenerate but means differ by {diff:.2e}"
    t = diff / se
    assert t < 4.0, (f"{metric}: benchmark {xb.mean():.4f} vs package {xp.mean():.4f}, "
                     f"t={t:.2f}")


@pytest.mark.parametrize("metric", ["welfare_all", "welfare_followers", "mean_actor_rate"])
def test_ranges_overlap(samples, metric):
    """A shifted support would mean the two engines explore different regimes."""
    b, p = samples
    xb = np.array([r[metric] for r in b], dtype=float)
    xp = np.array([r[metric] for r in p], dtype=float)
    slack = 1e-6            # the softmax epsilon, as above
    assert min(xb.max(), xp.max()) + slack > max(xb.min(), xp.min()), \
        f"{metric}: disjoint ranges"


def test_role_composition_is_comparable(samples):
    """Roles must be reachable on both sides in similar proportion — a broken
    step-1 or step-2 gate would show as one side never entering that role."""
    b, p = samples
    for role in ("n_reputation", "n_status"):
        fb = np.mean([r[role] > 0 for r in b])
        fp = np.mean([r[role] > 0 for r in p])
        assert abs(fb - fp) < 0.25, f"{role}: reached in {fb:.0%} vs {fp:.0%} of runs"
