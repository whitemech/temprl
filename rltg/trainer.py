import numpy as np
import shutil
from pprint import pprint

from gym import Env

from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.utils.Renderer import Renderer
from rltg.utils.StatsManager import StatsManager

import os

from rltg.utils.StoppingCondition import CheckAutomataInFinalState, GoalPercentage


class Trainer(object):
    def __init__(self, env:Env, agent:RLAgent=None, n_episodes=1000,
                 eval=False,
                 resume=False,
                 renderer:Renderer=None,
                 window_size=100,
                 stopping_conditions=(GoalPercentage(10, 1.0), CheckAutomataInFinalState()),
                 agent_data_dir="agent_data",
                 optimal_stats=1):
        assert agent is not None or resume
        self.env = env
        self.agent = agent if not resume else TGAgent.load(agent_data_dir)
        self.stopping_conditions = stopping_conditions
        self.n_episodes = n_episodes
        self.resume = resume
        self.eval = eval
        self.renderer = renderer
        self.window_size = window_size
        self.optimal_stats = optimal_stats

        if not self.resume:
            shutil.rmtree(agent_data_dir, ignore_errors=True)
            os.mkdir(agent_data_dir)

        self.agent_data_dir = agent_data_dir

        if self.eval and agent is not None:
            self.agent.set_eval(self.eval)

    def main(self):
        agent = self.agent

        num_episodes = self.n_episodes
        last_goal = False
        stats =         StatsManager(self.window_size)
        stats_optimal = StatsManager(self.window_size) if self.optimal_stats is not None else None

        if self.resume:
            agent.load(self.agent_data_dir)

        # Main training loop
        for ep in range(num_episodes):

            steps, total_reward, goal = self.train_loop()

            if not self.eval and ep % 100 == 0:
                agent.save(self.agent_data_dir)

            stats.update(steps, len(agent.brain.Q), total_reward, goal)
            stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.exploration_policy.epsilon, goal)

            if self.optimal_stats is not None and ep % self.optimal_stats == 0:
                agent.set_eval(True)
                steps, total_reward, goal = self.train_loop()
                stats_optimal.update(steps, len(agent.brain.Q), total_reward, goal)
                if goal:
                    pass
                    # print("Goal", total_reward, steps)
                agent.set_eval(False)
                agent.reset()

                if self.check_stop_conditions(agent, stats_optimal):
                    break

            # ids, table = list(zip(*sorted(agent.brain.Q.items(), key=lambda x: x[0])))
            # table = np.array(table)
            # print(ep)
            # print(table)
            #     pprint(agent.brain.Q)
            # stopping conditions


        agent.save(self.agent_data_dir)
        # stats.plot()

        return agent, stats, stats_optimal


    def train_loop(self):
        env = self.env
        agent = self.agent
        total_reward = 0
        steps = 0

        state = env.reset()
        if self.renderer:
            self.renderer.update(env)

        temporal_evaluators = agent.temporal_evaluators if isinstance(agent, TGAgent) else []

        stop_condition = False
        info = {"goal": False}

        # until the game is not ended and every temporal task is not failed or every temporal task is true
        while not stop_condition:
            action = agent.act(state)
            state2, reward, done, info = env.step(action)
            f_state, f_state2 = agent.sync(state, action, reward, state2)
            stop_condition = self.check_episode_stop_conditions(done, temporal_evaluators)
            agent.observe(state, action, reward, state2,
                          automata_states=f_state, automata_states2=f_state2, is_terminal_state=stop_condition)
            agent.replay()

            # add the observed reward (including the automaton reward)
            total_reward += agent.brain.obs_history[-1][2]
            steps += 1

            agent.update()
            state = state2

            if self.renderer:
                self.renderer.update(env)
        agent.reset()

        return steps, total_reward, info.get("goal", False) and all(t.is_true() for t in temporal_evaluators)

    def check_episode_stop_conditions(self, done, temporal_evaluators):
        any_te_failed = any(t.is_failed() for t in temporal_evaluators)
        all_te_true = all(t.is_true() for t in temporal_evaluators) if len(temporal_evaluators)>0 else False
        return done or all_te_true or any_te_failed

    def check_stop_conditions(self, agent, stats):
        temporal_evaluators = agent.temporal_evaluators if isinstance(agent, TGAgent) else []
        if not self.eval and all([s.check_condition(stats_manager=stats, temporal_evaluators=temporal_evaluators)
                                  for s in self.stopping_conditions]):
            return True

        return False
