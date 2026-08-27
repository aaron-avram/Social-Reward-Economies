"""
MultiAgentSystem — orchestration only.

This is the composition root: the ONE place that holds the whole SystemConfig and
the whole RngBundle. Everything below it receives narrow slices. If a method here
grows past ~30 lines it probably belongs in one of the other modules.
"""

from typing import Optional

import numpy as np

from model.agent import Agent, AgentRole
from model.config import SystemConfig, TrackingMode
from model.instrumentation import NullRecorder, Recorder, FullRecorder, role_update_diagnostic_row, checkpoint_bundle
from model.results import SimulationResults, StepRecord
from model.reputation import ReputationState, phase4, Phase4Trace, NO_LEADER
from model.rewards import build_reward_model
from model.rng import RngBundle
from model.roleupdate import RoleUpdateSchedule, update_roles_sequential
from model.welfare import true_reputation, current_policies, current_opinion_leader, paper_welfare

class MultiAgentSystem:

    def __init__(self, config: SystemConfig, rec: Recorder = NullRecorder()):
        """
        Constructs, in order:
            self.rng      = RngBundle(config.runtime.seed)
            self.rewards  = build_reward_model(config.reward, config.dims, self.rng.init)
            self.agents   = [Agent(i, config.algorithm, config.dims, self.rng.init) ...]
            self.rep      = ReputationState.initial(N)
            self.schedule = RoleUpdateSchedule(config.schedule)
            self.results  = SimulationResults()
            self.rec      = rec

        Agent construction order fixes RNG consumption order — keep it ascending.
        """
        self.config = config
        self.rec = rec
        self.time_step = 0

        N = config.dims.num_agents
        self.rng = RngBundle(config.runtime.seed)
        self.rewards = build_reward_model(config.reward, config.dims, self.rng.init)
        self.agents = [
            Agent(i, config.algorithm, config.dims, self.rng.init) for i in range(N)
        ]
        self.rep = ReputationState.initial(N)
        self.schedule = RoleUpdateSchedule(config.schedule)
        self.results = SimulationResults()

        # Consumed by the dense-history snapshot and by _track.
        self.last_active_actor_ids: np.ndarray = np.array([], dtype=int)
        self.last_active_participant_ids: np.ndarray = np.array([], dtype=int)
        self._last_observed_utility_matrix: Optional[np.ndarray] = None
        self._last_eta_v: float = 0.0
        self._last_phase4_trace: Optional[Phase4Trace] = None

    # ==================== per-step phases ====================

    def _sample_active_sets(self) -> tuple[np.ndarray, np.ndarray]:
        """
        PHASE 1. theta(mu) = 1 - exp(-mu) activation for actors and participants.

        CHANGE from the original (1917-1933): draw N uniforms per set per step and
        threshold, rather than drawing inside the loop. Makes each agent's draw a
        function of (seed, i, t) only, so common random numbers survive a parameter
        change. Costs a few unused draws; buys paired comparisons across the sweep.

        Returns sorted int arrays.
        """
        ids = np.arange(self.config.dims.num_agents)
        if self.config.runtime.force_all_active_debug:
            return ids.copy(), ids.copy()

        mu_a = np.array([a.state.actor_interaction_rate for a in self.agents])
        mu_p = np.array([a.state.participant_interaction_rate for a in self.agents])

        u_actor = self.rng.activation.random(size=ids.size)
        u_part = self.rng.activation.random(size=ids.size)

        actors = ids[u_actor < 1.0 - np.exp(-mu_a)]
        participants = ids[u_part < 1.0 - np.exp(-mu_p)]

        self.last_active_actor_ids = set(actors)
        self.last_active_participant_ids = set(participants)

        return actors, participants

    def _actors_act(self, actor_ids: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        """
        PHASE 2. Each active actor draws a state, picks an action, and every observer's
        utility for that (s, x) fills a column of the observed-utility matrix.

        Draw states and uniforms for ALL agents (same CRN reason as above), then use
        only the active ones. select_action takes the uniform (inverse-CDF).

        Returns (actions, payoffs, observed_utility_matrix).
        """
        N = self.config.dims.num_agents
        states = self.rng.action.integers(self.config.dims.num_states, size=N)
        uniforms = self.rng.action.random(size=N)

        leader_weights = self._leader_weights()

        actions: dict[int, tuple[int, int]] = {}
        payoffs: dict[int, float] = {}
        U = np.zeros((N, N), dtype=float)

        for k in actor_ids:
            k = int(k)
            agent = self.agents[k]
            s = int(states[k])
            x = agent.select_action(s, float(uniforms[k]), leader_weights)
            actions[k] = (s, x)

            observer_utilities = self.rewards.observer_utilities(s, x)
            U[:, k] = observer_utilities

            payoff = float(observer_utilities[k])
            payoffs[k] = payoff
            agent.state.payoff_history.append(payoff)
        
        self._last_observed_utility_matrix = U.copy()
        return actions, payoffs, U

    def _role_based_updates(self, actor_ids, actions, payoffs, U, sizes) -> None:
        """
        PHASE 3 (Section 6). Per-role learning for the active actors.

        Order within each actor matters: social_support_sum and the J^s update run
        BEFORE the role branch, because Step 2 of the role update compares
        kappa * J^s against J^pu and needs J^s current even for agents not yet in
        STATUS (1977-1982).
        """
        for k in actor_ids:
            k = int(k)
            agent = self.agents[k]
            state, action = actions[k]
            payoff = payoffs[k]

            # [STATUS-1] Social support is the followers' utility for THIS leader's
            # action — column k of the observed-utility matrix — not their own
            # actor payoffs. It is a SUM, not a mean (Eq. 11).
            social_support_sum = 0.0
            if agent.state.followers:
                social_support_sum = float(sum(
                    float(U[f, k]) for f in agent.state.followers
                ))
                # Keep J^s current before the agent formally switches into STATUS.
                if agent.state.role is not AgentRole.STATUS:
                    agent.state.estimated_reward_status += sizes.eta_J * (
                        social_support_sum - agent.state.estimated_reward_status
                    )

            if agent.state.role is AgentRole.PERSONAL_UTILITY:
                agent.update_personal_utility(
                    state, action, payoff, sizes.alpha_pu, sizes.eta_J
                )

            elif agent.state.role is AgentRole.REPUTATION:
                # [REP-6] J^r_i(t) is the current reputation estimate of the
                # followed agent, s_i(k,t) — not an EMA of the leader's payoff.
                if agent.state.following is not None:
                    agent.update_reputation_reward_estimate(
                        float(self.rep.s[k, agent.state.following])
                    )

            elif agent.state.role is AgentRole.STATUS:
                if agent.state.followers:
                    agent.update_status_optimization(
                        state, action, social_support_sum,
                        sizes.beta_status, sizes.eta_J
                    )

    def _leader_weights(self) -> dict[int, np.ndarray]:
        return {
            i: self.agents[a.state.following].get_behavior_weights()
            for i, a in enumerate(self.agents)
            if a.state.role is AgentRole.REPUTATION and a.state.following is not None
        }

    def _reputation_learning(self, U, actor_ids, participant_ids, sizes) -> None:
        """
        PHASE 4. One call now, not two implementations.

            self.rep, trace = phase4(self.rep, U, actor_ids, participant_ids,
                                     eta_v, self.config.algorithm, eq9_mode,
                                     leader_mode, self.rng.tiebreak,
                                     trace=self.rec.wants_phase4_trace)

        The actor-rate loop that was inside _phase4_updates_* (1608-1610) lives HERE,
        after the call: for each active participant, update_actor_interaction_rate.
        It is Eq. (13), not Eq. (9).
        """
        eta_v = sizes.eta_v

        self.rep, trace = phase4(
            self.rep,
            U,
            actor_ids,
            participant_ids,
            eta_v,
            self.config.algorithm,
            self.config.algorithm.eq9_averaging_mode,
            self.config.algorithm.leader_update_mode,
            self.rng.tiebreak,
            update_leader_estimates=True,
            trace=self.rec.wants_phase4_trace,
        )

        self._last_eta_v = float(eta_v)
        self._last_phase4_trace = trace
        self.rec.phase4(self.time_step, trace)

        # Eq. (13) — actor interaction rates. Was inside _phase4_updates_* at
        # 1608-1610; it is rate learning, not reputation learning.
        alpha_rate = sizes.alpha_rate
        for i in participant_ids:
            self.agents[int(i)].update_actor_interaction_rate(alpha_rate)


    def _adopt_leader_behavior(self) -> None:
        """
        PHASE 5. Resolve each REPUTATION agent's leader weights and pass them down —
        agents no longer hold a system reference.

        Read leaders from the pre-update follow graph so that within one step the
        order of iteration cannot matter.
        """
        leader_weights = self._leader_weights()
        for i, agent in enumerate(self.agents):
            if agent.state.role == AgentRole.REPUTATION:
                agent.adopt_behavior(leader_weights.get(i))

    def _maybe_update_roles(self) -> bool:
        """
        PHASE 6. self.schedule.is_due(t) -> update_roles_sequential(...).
        Returns whether an update fired.
        """
        fired = self.schedule.due_count(self.time_step)
        for _ in range(fired):
            update_roles_sequential(
                self.agents,
                self.rep.s,
                self.rep.L,
                self.config.algorithm,
                self.rng.order,
                t=self.time_step,
                rec=self.rec,
            )
        return fired > 0

    def step(self) -> None:
        """
        Should read as ~10 lines:

            t = (self.time_step := self.time_step + 1)
            sizes = self.config.stepsizes            # each .at(t) at point of use
            actors, participants = self._sample_active_sets()
            actions, payoffs, U = self._actors_act(actors)
            self._role_based_updates(actors, actions, payoffs, U, sizes)
            self._reputation_learning(U, actors, participants, sizes)
            self._adopt_leader_behavior()
            updated = self._maybe_update_roles()
            self._track(payoffs, len(actors), len(participants), updated)
        """
        self.time_step += 1
        sizes = self.config.stepsizes.at(self.time_step)
        actors, participants = self._sample_active_sets()
        actions, payoffs, U = self._actors_act(actors)
        self._role_based_updates(actors, actions, payoffs, U, sizes)
        self._reputation_learning(U, actors, participants, sizes)
        self._adopt_leader_behavior()
        updated = self._maybe_update_roles()
        self._track(payoffs, len(actors), len(participants), updated)

    # ==================== tracking + run ====================

    def _track(self, payoffs, num_actors, num_participants, role_updated: bool) -> None:
        """
        Build one StepRecord and append it. Always appends — `role_updated` only
        gates role_update_times and the diagnostic row (2412-2442).
        """
        rec = self._build_step_record(payoffs, num_actors, num_participants)
        self.results.append(rec, role_updated=role_updated)

        if role_updated and isinstance(self.rec, FullRecorder) and self.rec.role_update_diagnostics:
            index = len(self.results.role_update_times)
            tr = true_reputation(self.agents, self._policies(), self.rewards)
            self.results.role_update_diagnostics.append(
                role_update_diagnostic_row(
                    self.time_step, self.agents, self.rep.s, self.rep.L,
                    self.config.algorithm, index,
                )
            )
            bundle = checkpoint_bundle(
                self.time_step, self.agents, self.rep.s, self.rep.L, tr,
                self.config.algorithm, checkpoint_kind="role_update",
                role_update_index=index,
                eq9_mode=self.config.algorithm.eq9_averaging_mode,
                leader_mode=self.config.algorithm.leader_update_mode,
            )
            for key, rows in bundle.items():
                getattr(self.results, key).extend(rows)

    def _policies(self) -> np.ndarray:
        return current_policies(
            self.agents, self._leader_weights()
        )

    def _fill_dense(self, rec: StepRecord) -> None:
        """
        Dense per-timestep snapshots for small-N debug runs.
        Body from _record_small_n_trace_snapshot (811-855).

        The enable-flag check at 812-813 is gone — the caller already gated on
        self.rec.wants_dense_history.
        """
        N = self.config.dims.num_agents

        rec.dense_reputation = self.rep.s.copy()
        rec.dense_personal_benefit = self.rep.v.copy()
        rec.active_actor_ids = [int(i) for i in self.last_active_actor_ids]
        rec.active_participant_ids = [int(i) for i in self.last_active_participant_ids]
        rec.observed_utility_matrix = (
            None if self._last_observed_utility_matrix is None
            else self._last_observed_utility_matrix.copy()
        )
        rec.eta_v = float(self._last_eta_v)

        trace = self._last_phase4_trace
        if trace is None:
            rec.gossip_target_ids = []
            rec.averaging_agent_ids = []
            rec.avg_s_by_target = {}
            rec.delta_v_matrix = np.zeros((N, N), dtype=float)
        else:
            rec.gossip_target_ids = [int(k) for k in trace.gossip_target_ids]
            rec.averaging_agent_ids = [int(k) for k in trace.averaging_agent_ids]
            rec.avg_s_by_target = {int(k): float(v) for k, v in trace.avg_s_by_target.items()}
            rec.delta_v_matrix = (
                np.zeros((N, N), dtype=float) if trace.delta_v is None
                else trace.delta_v.copy()
            )

        tr = true_reputation(self.agents, self._policies(), self.rewards)
        rec.true_reputation = np.array(tr.true_reputation, dtype=float, copy=True)
        rec.true_reputation_rank = np.array(tr.true_rank, dtype=int, copy=True)
        rec.true_reputation_theta = np.array(tr.theta_mu, dtype=float, copy=True)
        rec.true_reputation_sum_expected = np.array(
            tr.sum_expected_utility_others, dtype=float, copy=True
        )

    def _build_step_record(self, payoffs, num_actors, num_participants) -> StepRecord:
        """Shared by _track and the async refresh path, so both see identical fields."""
        mode = self.config.runtime.tracking_mode
        full = mode is TrackingMode.FULL
        compact = full or self.rec.wants_compact_histories

        policies = self._policies()
        leader = current_opinion_leader(self.agents)
        w_all = paper_welfare(self.agents, policies, self.rewards,
                                      self.config.dims.num_states, leader_id=leader)
        w_fol = paper_welfare(self.agents, policies, self.rewards,
                                      self.config.dims.num_states, leader_id=leader,
                                      exclude_leader=True)

        roles = [a.state.role for a in self.agents]
        rec = StepRecord(
            t=self.time_step,
            follower_counts=[len(a.state.followers) for a in self.agents],
            actor_count=int(num_actors),
            participant_count=int(num_participants),
            online_active_actor_payoff_sum=float(sum(payoffs.values())),
            paper_welfare_all_agents=float(w_all),
            paper_welfare_followers_only=float(w_fol),
            status_count=sum(1 for r in roles if r is AgentRole.STATUS),
            pu_count=sum(1 for r in roles if r is AgentRole.PERSONAL_UTILITY),
            rep_count=sum(1 for r in roles if r is AgentRole.REPUTATION),
            role_label=[r.value for r in roles],          # always, even LIGHT (2447)
        )

        if compact:
            rec.estimated_reward_pu = [float(a.state.estimated_reward_pu) for a in self.agents]
            rec.estimated_reward_rep = [float(a.state.estimated_reward_rep) for a in self.agents]
            rec.estimated_reward_status = [float(a.state.estimated_reward_status) for a in self.agents]
            rec.actor_interaction_rate = [float(a.state.actor_interaction_rate) for a in self.agents]

            L = self.rep.L
            rec.highest_rep_agent = [int(L[i]) for i in range(len(self.agents))]
            rec.following = [-1 if a.state.following is None else int(a.state.following)
                             for a in self.agents]
            sel = [0.0 if L[i] == NO_LEADER else float(self.rep.s[i, L[i]])
                   for i in range(len(self.agents))]
            rec.selected_reputation = sel
            rec.weighted_selected_reputation = [self.config.algorithm.gamma * v for v in sel]

        if self.rec.wants_dense_history:
            self._fill_dense(rec)

        if full:
            weights = np.array([a.state.weights_pu.flatten() for a in self.agents])
            rec.norm_consensus = float(np.mean(np.var(weights, axis=0)))
            rec.expected_utilities = {
                i: float(np.mean(a.state.payoff_history)) if a.state.payoff_history else 0.0
                for i, a in enumerate(self.agents)
            }
            rec.actor_rates = [float(a.state.actor_interaction_rate) for a in self.agents]
            rec.roles = roles
            rec.actual_payoffs = dict(payoffs)

        return rec

    def simulate(self, num_steps: Optional[int] = None) -> SimulationResults:
        """
        Run the simulation and return results.

        Prints nothing — the progress and summary output at 2505-2535 moves to
        plots.summary_report(), so the three sweep harnesses stay quiet. A harness
        that wants the old output calls:
            print(plots.summary_report(results, cfg))
        """
        steps = self.config.runtime.num_time_steps if num_steps is None else int(num_steps)

        for _ in range(steps):
            self.step()

        self._finalize()
        return self.results

    def _finalize(self) -> None:
        """Run summary written once at the end (2536-2538)."""
        follower_counts = [len(a.state.followers) for a in self.agents]
        self.results.final_roles = [a.state.role for a in self.agents]
        self.results.final_followers = follower_counts
        self.results.opinion_leader = (
            int(np.argmax(follower_counts)) if max(follower_counts, default=0) > 0 else -1
        )