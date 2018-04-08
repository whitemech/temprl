import numpy as np
from gym import Env

from rltg.agents.RLAgent import RLAgent
from rltg.utils.Renderer import Renderer
from rltg.utils.StatsManager import StatsManager


def goal_perc_threshold(*args, **kwargs):
    goal_percentage = kwargs["goal_percentage"]
    return goal_percentage >= 97.00


ID2ACTION = {0: 2, 1: 3}
class Trainer(object):
    def __init__(self, env:Env, agent:RLAgent, stopping_conditions=[goal_perc_threshold], n_episodes=1000, eval=False, resume=False, renderer:Renderer=None):
        self.env = env
        self.agent = agent
        self.stopping_conditions = stopping_conditions
        self.n_episodes = n_episodes
        self.resume = resume
        self.eval = eval
        self.renderer = renderer

        if self.eval:
            self.agent.set_eval(self.eval)

    def main(self):
        env = self.env
        agent = self.agent
        num_episodes = self.n_episodes
        steps = 0

        stats = StatsManager()

        if self.resume:
            agent.load("data/agent_data")

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
                    self.renderer.update(env.render())

            stats.update(len(agent.brain.Q), total_reward, info["goal"])
            stats.print_summary(ep, steps, len(agent.brain.Q), total_reward, agent.exploration_policy.epsilon, info["goal"])

            # stopping conditions
            if all([s(goal_percentage=np.mean(stats.goals[-100:])*100) if len(stats.goals)>=100 else False for s in self.stopping_conditions]):
                break

            agent.reset()
            if ep%100==0:
                agent.save("data/agent_data")


        stats.plot()
