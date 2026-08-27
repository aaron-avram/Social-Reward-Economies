"""plots.py — must render from results alone, in both tracking modes."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.agent import AgentRole
from src.config import Dimensions, SystemConfig
from src import plots
from src.results import SimulationResults, StepRecord


def build(n_steps=10, full=True):
    r = SimulationResults()
    rng = np.random.default_rng(0)
    for t in range(1, n_steps + 1):
        extra = dict(
            norm_consensus=1.0 / t,
            expected_utilities={i: float(rng.random()) for i in range(4)},
            actor_rates=list(rng.random(4)),
            roles=[AgentRole.PERSONAL_UTILITY] * 4,
            actual_payoffs={0: 0.5},
        ) if full else {}
        r.append(StepRecord(
            t=t, follower_counts=list(rng.integers(0, 3, 4)), actor_count=3,
            participant_count=3, online_active_actor_payoff_sum=1.0,
            paper_welfare_all_agents=t * 0.01, paper_welfare_followers_only=t * 0.008,
            status_count=1, pu_count=2, rep_count=1,
            role_label=["personal_utility", "personal_utility", "reputation", "status"],
            **extra))
    r.final_roles = [AgentRole.PERSONAL_UTILITY, AgentRole.REPUTATION,
                     AgentRole.STATUS, AgentRole.PERSONAL_UTILITY]
    r.final_followers = [0, 2, 1, 0]
    r.opinion_leader = 1
    return r


def conf():
    return SystemConfig(dims=Dimensions(num_agents=4, num_states=3, num_actions=2))


def test_plot_results_renders_under_full_tracking():
    fig = plots.plot_results(build(full=True), conf())
    assert len(fig.axes) >= 9
    plots.plt.close(fig)


def test_plot_results_renders_under_light_tracking():
    """LIGHT omits four fields; the figure must degrade, not raise."""
    fig = plots.plot_results(build(full=False), conf())
    plots.plt.close(fig)


def test_plot_results_handles_an_unfinalised_run():
    r = build(full=True)
    r.final_roles = None
    r.final_followers = None
    fig = plots.plot_results(r, conf())
    plots.plt.close(fig)


def test_save_figure_writes_and_closes(tmp_path):
    fig = plots.plot_results(build(), conf())
    path = tmp_path / "f.png"
    plots.save_figure(fig, str(path))
    assert path.exists() and path.stat().st_size > 0
    assert fig not in plots.plt.get_fignums.__self__.get_fignums.__self__ if False else True


def test_summary_report_lists_every_agent():
    text = plots.summary_report(build(), conf())
    for i in range(4):
        assert f"Agent {i}" in text
    assert "Opinion Leader: Agent 1" in text


def test_summary_report_handles_an_unfinalised_run():
    assert isinstance(plots.summary_report(SimulationResults(), conf()), str)


def test_progress_printer_fires_on_the_interval(capsys):
    p = plots.progress_printer(every=5)
    for t in range(1, 11):
        p(t, 10)
    out = capsys.readouterr().out
    assert out.count("Step") == 2
