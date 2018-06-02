import logging
import os
import shutil
import time

from rltg.agents.Agent import Agent
from rltg.trainers.Trainer import DEFAULT_DATA_DIR, Trainer
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper
from rltg.utils.StatsManager import StatsManager
from rltg.utils.StoppingCondition import GoalPercentage

import pickle

from rltg.utils.misc import logger_from_verbosity


class GenericTrainer(Trainer):

    def __init__(self, env:GoalEnvWrapper, agent:Agent, n_episodes=1000,
                 data_dir=DEFAULT_DATA_DIR,
                 stop_conditions=(GoalPercentage(10, 1.0), ),):
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


    def main(self, eval:bool=False, render:bool=False, verbosity:int=1):
        logger_from_verbosity(verbosity)

        agent = self.agent
        num_episodes = self.n_episodes
        stats = self.stats
        optimal_stats = self.optimal_stats

        for ep in range(self.cur_episode, num_episodes):

            if not eval:
                steps, total_reward, goal = self.train_loop(render=render)
                stats.update(steps, len(agent.brain.Q), total_reward, goal)
                summary = stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.brain.policy.epsilon.get(), goal)
                logging.info(summary)

            # try optimal run
            agent.set_eval(True)
            steps, total_reward, goal = self.train_loop(render=render)
            optimal_stats.update(steps, len(agent.brain.Q), total_reward, goal)
            optimal_summary = optimal_stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.brain.policy.epsilon.get(), goal)
            logging.info(optimal_summary + " * optimal * ")
            agent.set_eval(False)


            if self.check_stop_conditions(optimal_stats, eval=eval):
                break

            if self.cur_episode%100==0:
                agent.save(self.agent_data_dir)
                self.save()

            self.cur_episode = ep

        if not eval:
            self.save()
            agent.save(self.agent_data_dir)

        stats.to_csv(self.data_dir + "/" + stats.name + "_" + str(time.time()))
        optimal_stats.to_csv(self.data_dir + "/" + optimal_stats.name + "_" + str(time.time()))

        return stats, optimal_stats

    def train_loop(self, render:bool=False):
        env = self.env
        agent = self.agent

        stop_condition = False
        info = {"goal": False}

        state = env.reset()
        action = agent.start(state)
        obs = None
        if render: env.render()

        while not stop_condition:
            state2, reward, done, info = env.step(action)
            if render: env.render()

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
        self.stats.reset()
        self.optimal_stats.reset()
