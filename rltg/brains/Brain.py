from abc import ABC, abstractmethod

from gym.core import Space


class Brain(ABC):
    def __init__(self, state_space:Space, action_space:Space):
        self.state_space = state_space
        self.action_space = action_space

    @abstractmethod
    def best_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    @abstractmethod
    def observe(self, *args):
        raise NotImplementedError

    def reset(self):
        """action performed at the end of each episode"""
        raise NotImplementedError
