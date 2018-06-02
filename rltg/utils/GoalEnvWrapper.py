import gym
from gym import Wrapper
from gym.utils import EzPickle


class GoalEnvWrapper(Wrapper, EzPickle):
    """A OpenAI Gym environment goal-based."""

    def __init__(self, env, f):
        Wrapper.__init__(self, env)
        EzPickle.__init__(self, env, f)
        self.f = f

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["goal"] = self.f(obs, reward, done, info)
        return obs, reward, done, info
