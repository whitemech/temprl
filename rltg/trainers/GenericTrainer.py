import logging
import os
import shutil
import time
from abc import ABC

from rltg.agents.Agent import Agent
from rltg.agents.RLAgent import RLAgent
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper
from rltg.utils.Renderer import Renderer
from rltg.utils.StatsManager import StatsManager
from rltg.utils.StoppingCondition import GoalPercentage

import pickle

DEFAULT_DATA_DIR = "data"
DEFAULT_TRAINER_FILEPATH = DEFAULT_DATA_DIR + "/" + "trainer.pkl"

class GenericTrainer(ABC):

    def __init__(self, env:GoalEnvWrapper, agent:Agent, n_episodes=1000,
                 data_dir=DEFAULT_DATA_DIR,
                 stop_conditions=(GoalPercentage(10, 1.0), )):
        self.env = env
        self.n_episodes = n_episodes
        self.stop_conditions = list(stop_conditions)

        self.data_dir = data_dir
        self.agent_data_dir = data_dir + "/agent_data"

        self.agent = agent

        shutil.rmtree(self.data_dir, ignore_errors=True)
        os.mkdir(self.data_dir)
        os.mkdir(self.agent_data_dir)

        self.cur_episode = 0
        self.stats = StatsManager(name="train_stats")
        self.optimal_stats = StatsManager(name="eval_stats")


    def main(self, eval:bool=False, renderer:Renderer=None):
        agent = self.agent
        num_episodes = self.n_episodes
        stats = self.stats
        optimal_stats = self.optimal_stats

        for ep in range(self.cur_episode, num_episodes):

            if not eval:
                steps, total_reward, goal = self.train_loop(renderer=renderer)
                stats.update(steps, len(agent.brain.Q), total_reward, goal)
                stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.brain.policy.epsilon.get(), goal)

            # try optimal run
            agent.set_eval(True)
            steps, total_reward, goal = self.train_loop(renderer=renderer)
            optimal_stats.update(steps, len(agent.brain.Q), total_reward, goal)

            print("Try optimal run:")
            optimal_stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.brain.policy.epsilon.get(), goal)
            agent.set_eval(False)


            if self.check_stop_conditions(optimal_stats, eval=eval):
                break

            if self.cur_episode%100==0:
                agent.save(self.agent_data_dir)
                self.save()

            self.cur_episode = ep


        self.save()
        agent.save(self.agent_data_dir)
        stats.to_csv(self.data_dir + "/" + stats.name + "_" + str(time.time()))
        optimal_stats.to_csv(self.data_dir + "/" + optimal_stats.name + "_" + str(time.time()))

        return stats, optimal_stats

    def train_loop(self, renderer=None):
        env = self.env
        agent = self.agent

        stop_condition = False
        info = {"goal": False}

        state = env.reset()
        action = agent.start(state)
        obs = None
        if renderer is not None: renderer.update(env)

        while not stop_condition:
            state2, reward, done, info = env.step(action)
            if renderer is not None: renderer.update(env)

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

    def check_stop_conditions(self, stats:StatsManager, *args, eval=False, **kwargs):
        return not eval and all([s.check_condition(stats_manager=stats) for s in self.stop_conditions])

    def check_episode_stop_conditions(self, done, goal, *args, **kwargs):
        return done or goal

    def is_goal(self, info, *args, **kwargs):
        return info.get("goal", False)

    def save(self):
        with open(self.data_dir + "/trainer.pkl", "wb") as fout:
            pickle.dump(self, fout)

    def reset(self):
        self.cur_episode = 0


    @staticmethod
    def load(filepath=DEFAULT_TRAINER_FILEPATH):
        with open(filepath, "rb") as fin:
            return pickle.load(fin)

    @staticmethod
    def resume(filepath=DEFAULT_TRAINER_FILEPATH):
        trainer = GenericTrainer.load(filepath)
        trainer.main()

    @staticmethod
    def eval(filepath=DEFAULT_TRAINER_FILEPATH):
        trainer = GenericTrainer.load(filepath)
        trainer.main(eval=True)
