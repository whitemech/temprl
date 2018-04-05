from abc import ABC, abstractmethod

from gym.core import Space


class Brain(ABC):
    def __init__(self, observation_space:Space, action_space:Space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.episode = 0
        self.iteration = 0
        self.episode_iteration = 0

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    @abstractmethod
    def observe(self, *args):
        raise NotImplementedError

    def reset(self):
        """action performed at the end of each episode"""
        self.episode += 1
        self.episode_iteration = 0

    def update(self):
        """action performed at the end of each iteration"""
        self.episode_iteration += 1
        self.iteration += 1

