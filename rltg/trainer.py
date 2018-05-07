import shutil

from gym import Env

from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.utils.Renderer import Renderer
from rltg.utils.StatsManager import StatsManager

import os

from rltg.utils.StoppingCondition import CheckAutomataInFinalState, GoalPercentage


class Trainer(object):
    def __init__(self, env:Env, agent:RLAgent, n_episodes=1000,
                 eval=False,
                 resume=False,
                 try_optimal_run=True,
                 renderer:Renderer=None,
                 window_size=100,
                 stopping_conditions=(GoalPercentage(10, 1.0), CheckAutomataInFinalState()),
                 agent_data_dir="agent_data"):
        self.env = env
        self.agent = agent
        self.stopping_conditions = stopping_conditions
        self.n_episodes = n_episodes
        self.resume = resume
        self.eval = eval
        self.try_optimal_run = try_optimal_run
        self.renderer = renderer
        self.window_size = window_size

        if not self.resume:
            shutil.rmtree(agent_data_dir, ignore_errors=True)
            os.mkdir(agent_data_dir)

        self.agent_data_dir = agent_data_dir

        if self.eval:
            self.agent.set_eval(self.eval)

    def main(self):
        agent = self.agent

        num_episodes = self.n_episodes
        last_goal = False
        stats = StatsManager(self.window_size)

        if self.resume:
            agent.load(self.agent_data_dir)

        # Main training loop
        for ep in range(num_episodes):

            # switch between training mode and evaluation mode
            # to check if policy reached is optimal.
            # only when in training mode
            steps, total_reward, goal = self.train_loop(try_optimal=self.eval or (last_goal and self.try_optimal_run))
            last_goal = goal

            stats.update(len(agent.brain.Q), total_reward, goal)
            stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.exploration_policy.epsilon, goal)

            # stopping conditions
            if self.check_stop_conditions(agent, stats):
                break

            agent.reset()
            if not self.eval and ep % 100 == 0:
                agent.save(self.agent_data_dir)

        agent.save(self.agent_data_dir)
        stats.plot()


    def train_loop(self, try_optimal=False):
        env = self.env
        agent = self.agent
        total_reward = 0
        steps = 0

        state = env.reset()
        if self.renderer:
            self.renderer.update(env)

        temporal_evaluators = agent.temporal_evaluators if isinstance(agent, TGAgent) else []

        done, all_te_not_failed, all_te_true = False, True, False
        info = {"goal": False}

        # until the game is not ended and every temporal task is not failed or every temporal task is true
        while not done and all_te_not_failed and not all_te_true:
            action = agent.act(state, best_action=try_optimal)
            state2, reward, done, info = env.step(action)
            agent.observe(state, action, reward, state2)
            agent.replay()

            # add the observed reward (including the automaton reward)
            total_reward += agent.brain.obs_history[-1][2]
            steps += 1

            agent.update()
            state = state2

            if self.renderer:
                self.renderer.update(env)

            all_te_not_failed = all(not t.is_failed() for t in temporal_evaluators)
            all_te_true   = all(t.is_true() for t in temporal_evaluators)

        return steps, total_reward, info["goal"] and all(t.is_true() for t in temporal_evaluators)

    def check_stop_conditions(self, agent, stats):
        temporal_evaluators = agent.temporal_evaluators if isinstance(agent, TGAgent) else []
        if not self.eval and all([s.check_condition(stats_manager=stats, temporal_evaluators=temporal_evaluators)
                                  for s in self.stopping_conditions]):
            return True

        return False
