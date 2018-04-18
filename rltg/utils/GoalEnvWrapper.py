import gym
from gym import Wrapper


class GoalEnvWrapper(Wrapper):
    def __init__(self, env, f):
        super().__init__(env)
        self.f = f

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["goal"] = self.f(obs, reward, done, info)
        return obs, reward, done, info
