"""
Presentation. Depends on results.py, config.py, and agent.py only.

matplotlib is imported HERE and nowhere else. Today importing the engine drags it
in and all three sweep harnesses pay for it.

Every function takes SimulationResults and returns a Figure, so plots can be
regenerated from a saved .npz without re-running the simulation. Nothing here
touches a MultiAgentSystem.

Two changes from plot_results (2543-2676):
  * It is nine panels, not six. Split one function per panel; the original is 134
    lines in one method and the panels share nothing but the GridSpec.
  * Saving and printing are the caller's business. plot_results() returns a Figure;
    savefig lives in save_figure(), and the "Plot saved to ..." print (2676) is gone.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from model.agent import AgentRole
from model.config import SystemConfig
from model.results import SimulationResults

_ROLE_ORDER = (
    (AgentRole.PERSONAL_UTILITY, "Personal Utility", "#ff9999"),
    (AgentRole.REPUTATION, "Reputation", "#66b3ff"),
    (AgentRole.STATUS, "Status", "#99ff99"),
)

# 1=PU, 2=Rep, 3=Status, matching the colourbar label at 2594.
_ROLE_CODE = {
    AgentRole.PERSONAL_UTILITY: 1,
    AgentRole.REPUTATION: 2,
    AgentRole.STATUS: 3,
}


def _agent_colors(num_agents: int) -> np.ndarray:
    """Per-agent line colours, shared across panels 2, 3 and 6 (2563)."""
    return plt.cm.tab10(np.linspace(0, 1, num_agents))


def _style(ax: Axes, title: str, xlabel: str = "", ylabel: str = "",
           legend: bool = False, grid_axis: str = "both") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis=grid_axis)


# ============================================================================
# Panels
# ============================================================================

def panel_norm_consensus(ax: Axes, results: SimulationResults) -> None:
    """Policy-weight variance on a log axis. FULL tracking only (2550-2555)."""
    if not results.norm_consensus:
        _empty(ax, "Norm Convergence", "requires tracking_mode=FULL")
        return
    ax.semilogy(results.norm_consensus, linewidth=2, color="darkblue")
    _style(ax, "Norm Convergence (Policy Weight Variance)", "Timestep",
           "Variance (log scale)")


def panel_expected_utilities(ax: Axes, results: SimulationResults,
                             num_agents: int) -> None:
    """
    Per-agent mean payoff over the trajectory. FULL only (2558-2570).

    expected_utilities is a list of dicts keyed by agent id, not a rectangular
    array, so it is densified here.
    """
    if not results.expected_utilities:
        _empty(ax, "Utility Learning", "requires tracking_mode=FULL")
        return
    utils = np.zeros((len(results.expected_utilities), num_agents))
    for t, row in enumerate(results.expected_utilities):
        for agent_id, value in row.items():
            utils[t, int(agent_id)] = value
    colors = _agent_colors(num_agents)
    for i in range(num_agents):
        ax.plot(utils[:, i], label=f"Agent {i}", color=colors[i],
                alpha=0.8, linewidth=1.5)
    _style(ax, "Utility Learning (Section 6.3, 6.5)", "Timestep",
           "Expected Utility", legend=True)


def panel_follower_counts(ax: Axes, results: SimulationResults,
                          num_agents: int) -> None:
    """Follower count per agent over time (2573-2581). Both tracking modes."""
    if not results.follower_counts:
        _empty(ax, "Opinion Leader Emergence", "no data")
        return
    followers = np.array(results.follower_counts)
    colors = _agent_colors(num_agents)
    for i in range(num_agents):
        ax.plot(followers[:, i], label=f"Agent {i}", linewidth=2, color=colors[i])
    _style(ax, "Opinion Leader Emergence (Section 7)", "Timestep",
           "Follower Count", legend=True)


def panel_role_evolution(ax: Axes, results: SimulationResults) -> None:
    """
    Role heatmap, agents on rows and time on columns (2584-2594).

    Reads role_label_history (strings, written in BOTH modes at 2447) rather than
    roles_history (enums, FULL only at 2500), so this panel works under LIGHT.
    """
    if not results.role_label_history:
        _empty(ax, "Role Evolution", "no data")
        return
    code_by_label = {role.value: code for role, code in _ROLE_CODE.items()}
    codes = np.array([[code_by_label[label] for label in row]
                      for row in results.role_label_history])
    im = ax.imshow(codes.T, aspect="auto", cmap="viridis", interpolation="nearest")
    _style(ax, "Role Evolution (Section 7)", "Timestep", "Agent ID")
    ax.grid(False)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Role: 1=PU, 2=Rep, 3=Status")


def panel_active_sets(ax: Axes, results: SimulationResults) -> None:
    """|A_a(t)| and |A_p(t)| (2597-2605)."""
    ax.plot(results.actor_counts, label="Actors |A_a(t)|",
            color="teal", linewidth=2)
    ax.plot(results.participant_counts, label="Participants |A_p(t)|",
            color="orange", linewidth=2)
    _style(ax, "Active Actor/Participant Sets (Section 6)", "Timestep",
           "Count", legend=True)


def panel_actor_rates(ax: Axes, results: SimulationResults,
                      num_agents: int, budget_M: float) -> None:
    """
    Learned mu_{a,i}(t) (2608-2617). FULL only.

    NOTE actor_interaction_rate_history holds the same values and is written under
    compact debug too — switch to it if you want this panel under LIGHT.
    """
    if not results.actor_rates:
        _empty(ax, "Learned Actor Rates", "requires tracking_mode=FULL")
        return
    rates = np.array(results.actor_rates)
    colors = _agent_colors(num_agents)
    for i in range(num_agents):
        ax.plot(rates[:, i], label=f"Agent {i}", color=colors[i], linewidth=1.5)
    ax.set_ylim([0, budget_M * 1.1])
    _style(ax, r"Learned Actor Rates $\mu_{a,i}(t)$ (Eq. 13)", "Timestep",
           "Actor Rate", legend=True)


def panel_welfare(ax: Axes, results: SimulationResults) -> None:
    """
    Both paper welfare series (2620-2639).

    The `.get(key, [])` guards at 2621 and 2628 are gone — these fields always
    exist on SimulationResults. The emptiness checks remain, since a zero-step
    run has nothing to draw.
    """
    if results.paper_welfare_followers_only:
        ax.plot(results.paper_welfare_followers_only, linewidth=2,
                color="darkgreen", label="Followers-only paper welfare")
    if results.paper_welfare_all_agents:
        ax.plot(results.paper_welfare_all_agents, linewidth=1.8,
                color="steelblue", label="All-agents paper welfare")
    _style(ax, "Paper Welfare Over Time", "Timestep", "Expected Welfare",
           legend=True)


def panel_final_roles(ax: Axes, results: SimulationResults) -> None:
    """Final role distribution (2642-2653). Needs _finalize() to have run."""
    if results.final_roles is None:
        _empty(ax, "Final Role Distribution", "run finalize() first")
        return
    labels = [label for _, label, _ in _ROLE_ORDER]
    counts = [sum(1 for r in results.final_roles if r is role)
              for role, _, _ in _ROLE_ORDER]
    colors = [color for _, _, color in _ROLE_ORDER]
    ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=2)
    _style(ax, "Final Role Distribution", ylabel="Count", grid_axis="y")


def panel_final_followers(ax: Axes, results: SimulationResults) -> None:
    """Final follower distribution, leader highlighted (2656-2667)."""
    if results.final_followers is None:
        _empty(ax, "Final Follower Distribution", "run finalize() first")
        return
    counts = results.final_followers
    leader = results.opinion_leader
    colors = ["#ff6666" if i == leader else "#6666ff" for i in range(len(counts))]
    ax.bar(range(len(counts)), counts, color=colors, edgecolor="black", linewidth=2)
    ax.set_xticks(range(len(counts)))
    _style(ax, f"Final Follower Distribution (Opinion Leader: Agent {leader})",
           "Agent ID", "Follower Count", grid_axis="y")


def _empty(ax: Axes, title: str, reason: str) -> None:
    """Placeholder for a panel whose data was not collected, so a LIGHT-mode run
    still produces a readable figure instead of raising."""
    ax.text(0.5, 0.5, reason, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="grey")
    ax.set_xticks([])
    ax.set_yticks([])
    _style(ax, title)


# ============================================================================
# Figure
# ============================================================================

def plot_results(results: SimulationResults, config: SystemConfig) -> Figure:
    """
    The nine-panel summary grid. Body from plot_results (2543-2675).

    Returns the Figure; the caller decides whether to save or show it. Panels
    whose fields were not collected under the run's tracking mode render a
    placeholder rather than raising.
    """
    num_agents = config.dims.num_agents

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig)

    panel_norm_consensus(fig.add_subplot(gs[0, 0]), results)
    panel_expected_utilities(fig.add_subplot(gs[0, 1]), results, num_agents)
    panel_follower_counts(fig.add_subplot(gs[0, 2]), results, num_agents)
    panel_role_evolution(fig.add_subplot(gs[1, :2]), results)
    panel_active_sets(fig.add_subplot(gs[1, 2]), results)
    panel_actor_rates(fig.add_subplot(gs[2, 0]), results, num_agents,
                      float(config.algorithm.M))
    panel_welfare(fig.add_subplot(gs[2, 1]), results)
    panel_final_roles(fig.add_subplot(gs[2, 2]), results)
    panel_final_followers(fig.add_subplot(gs[3, :]), results)

    fig.suptitle(
        "Sections 6-7: Corrected Learning Algorithms\n"
        "Personal Utility + Reputation Learning + Status Optimization + "
        "Actor Rates + Sequential Role Updates",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str, *, dpi: int = 150) -> None:
    """Save and close. Closing matters in a sweep — matplotlib keeps every open
    figure alive and a few hundred will exhaust memory."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Text summary
# ============================================================================

def summary_report(results: SimulationResults, config: SystemConfig) -> str:
    """
    The end-of-run text that simulate() used to print (2512-2535). Returns a
    string so the caller decides whether it reaches stdout.

    Expected utilities come from the LAST tracked step's per-agent means, not from
    agent.state.payoff_history — this function has no access to agents. Under
    LIGHT tracking that field is absent and the section is omitted.
    """
    lines: list[str] = ["", "=" * 70, "FINAL RESULTS", "=" * 70]

    if results.final_roles is None or results.final_followers is None:
        lines.append("\n(run finalize() before requesting a summary)")
        return "\n".join(lines)

    lines.append("\nFinal Roles:")
    for i, role in enumerate(results.final_roles):
        followers = results.final_followers[i]
        lines.append(f"  Agent {i}: {role.value:20s} (followers: {followers})")

    leader = results.opinion_leader
    if leader >= 0:
        lines.append(
            f"\nOpinion Leader: Agent {leader} "
            f"with {results.final_followers[leader]} followers"
        )

    if results.expected_utilities:
        lines.append("\nExpected Utilities (mean payoff over trajectory):")
        final = results.expected_utilities[-1]
        for i in sorted(final):
            lines.append(f"  Agent {i}: {final[i]:.4f}")

    rates = results.actor_rates or results.actor_interaction_rate_history
    if rates:
        lines.append("\nFinal Actor Interaction Rates (learned):")
        for i, rate in enumerate(rates[-1]):
            lines.append(f"  Agent {i}: {rate:.4f}")

    return "\n".join(lines)


def progress_printer(every: int = 500):
    """
    Drop-in for simulate(on_step=...) reproducing the progress output at 2508-2509.
    Sweeps pass nothing and stay quiet.
    """
    def _on_step(t: int, total: int) -> None:
        if t % every == 0:
            print(f"  Step {t}/{total}")
    return _on_step
