import shutil

import numpy as np
from gym import Env

from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.utils.Renderer import PixelRenderer
from rltg.utils.StatsManager import StatsManager

import os

def goal_perc_threshold(*args, **kwargs)->bool:
    goal_history = kwargs["goal_history"]
    if len(goal_history)<100:
        return False
    goal_percentage = np.mean(goal_history[-100:])*100
    return goal_percentage >= 97.00

def check_automata_in_final_state(*args, **kwargs)->bool:
    temporal_evaluators = kwargs["temporal_evaluators"]
    # all the automata has to be in their final state
    return all(t.simulator.is_true() for t in temporal_evaluators)



ID2ACTION = {0: 2, 1: 3}
class Trainer(object):
    def __init__(self, env:Env, agent:RLAgent, n_episodes=1000,
                 eval=False,
                 resume=False,
                 renderer:PixelRenderer=None,
                 stopping_conditions=(goal_perc_threshold, check_automata_in_final_state),
                 agent_data_dir="agent_data"):
        self.env = env
        self.agent = agent
        self.stopping_conditions = stopping_conditions
        self.n_episodes = n_episodes
        self.resume = resume
        self.eval = eval
        self.renderer = renderer

        if not self.resume:
            shutil.rmtree(agent_data_dir, ignore_errors=True)
            os.mkdir(agent_data_dir)

        self.agent_data_dir = agent_data_dir

        if self.eval:
            self.agent.set_eval(self.eval)

    def main(self):
        env = self.env
        agent = self.agent
        num_episodes = self.n_episodes
        steps = 0

        stats = StatsManager()

        if self.resume:
            agent.load(self.agent_data_dir)

        # Main training loop
        for ep in range(num_episodes):

            total_reward = 0
            steps = 0

            state = env.reset()
            done = False

            while not done:
                action = agent.act(state)
                state2, reward, done, info = env.step(action)
                agent.observe(state, action, reward, state2)
                agent.replay()

                # add the observed reward (including the automaton reward)
                try:
                    total_reward += agent.brain.obs_history[-1][2]
                except:
                    total_reward += reward


                state = state2

                steps += 1
                if done:
                    break

                agent.update()
                if self.renderer:
                    self.renderer.update(env)

            stats.update(len(agent.brain.Q), total_reward, info["goal"])
            stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.exploration_policy.epsilon, info["goal"])

            # stopping conditions
            temporal_evaluators = agent.temporal_evaluators if isinstance(agent, TGAgent) else []
            if all([s(goal_history=stats.goals, temporal_evaluators=temporal_evaluators)
                    for s in self.stopping_conditions]):
                break

            agent.reset()
            if ep%100==0:
                agent.save(self.agent_data_dir)

        agent.save(self.agent_data_dir)
        stats.plot()
