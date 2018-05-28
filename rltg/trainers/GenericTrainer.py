import logging
import os
import shutil
import time
from abc import ABC

from gym import Env

from rltg.agents.Agent import Agent
from rltg.agents.RLAgent import RLAgent
from rltg.utils.Renderer import Renderer
from rltg.utils.StatsManager import StatsManager
from rltg.utils.StoppingCondition import GoalPercentage


class GenericTrainer(ABC):

    def __init__(self, env:Env, agent:Agent=None, n_episodes=1000,
                 resume=False,
                 eval=False,
                 data_dir="data",
                 stop_conditions=(GoalPercentage(10, 1.0), ),
                 renderer: Renderer = None, ):
        self.env = env
        self.n_episodes = n_episodes
        self.stop_conditions = list(stop_conditions)
        self.renderer = renderer

        self.eval = eval
        self.resume = resume
        self.data_dir = data_dir
        self.agent_data_dir = data_dir + "/agent_data"

        self.agent = agent if not resume else self.load_agent()
        if not self.resume:
            shutil.rmtree(self.data_dir, ignore_errors=True)
            os.mkdir(self.data_dir)
            os.mkdir(self.agent_data_dir)

        self.agent.set_eval(self.eval)

    def main(self):

        logging.info("Start training")

        agent = self.agent
        num_episodes = self.n_episodes
        stats = StatsManager(name="train_stats")
        optimal_stats = StatsManager(name="eval_stats")

        for ep in range(num_episodes):

            steps, total_reward, goal = self.train_loop()
            stats.update(steps, len(agent.brain.Q), total_reward, goal)
            stats.print_summary(agent.brain.episode, steps, len(agent.brain.Q), total_reward, agent.brain.policy.epsilon.get(), goal)

            # try optimal run
            agent.set_eval(True)
            steps, total_reward, goal = self.train_loop()
            optimal_stats.update(steps, len(agent.brain.Q), total_reward, goal)
            optimal_stats.print_summary(agent.brain.episode, steps, len(agent.brain.Q), total_reward, agent.brain.policy.epsilon.get(), goal)
            agent.set_eval(False)


            if self.check_stop_conditions(optimal_stats):
                break

            if ep%100==0:
                agent.save(self.agent_data_dir)


        agent.save(self.agent_data_dir)
        stats.to_csv(self.data_dir + "/" + stats.name + "_" + str(time.time()))
        optimal_stats.to_csv(self.data_dir + "/" + optimal_stats.name + "_" + str(time.time()))

        return stats, optimal_stats

    def train_loop(self):
        env = self.env
        agent = self.agent

        stop_condition = False
        info = {"goal": False}

        state = env.reset()
        action = agent.start(state)
        obs = None
        if self.renderer: self.renderer.update(env)

        while not stop_condition:
            state2, reward, done, info = env.step(action)
            if self.renderer: self.renderer.update(env)

            stop_condition = self.check_episode_stop_conditions(done, info.get("goal", False))
            obs = agent.observe(state, action, reward, state2, is_terminal_state=stop_condition)

            if stop_condition:
                break

            action = agent.step(obs)
            agent.update()
            state = state2

        agent.end(obs)
        steps, tot_reward = agent.brain.episode_iteration, agent.brain.total_reward
        return steps, tot_reward, self.is_goal(info)

    def check_stop_conditions(self, stats:StatsManager, *args, **kwargs):
        return all([s.check_condition(stats_manager=stats) for s in self.stop_conditions])

    def check_episode_stop_conditions(self, done, goal, *args, **kwargs):
        return done or goal

    def is_goal(self, info, *args, **kwargs):
        return info.get("goal", False)

    def load_agent(self):
        return RLAgent.load(self.agent_data_dir)

