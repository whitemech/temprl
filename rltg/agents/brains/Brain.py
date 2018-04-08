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
        """From a state, return the action for the implemented approach.
        e.g. in Q-Learning, select the argmax of the Q-values relative to the 'state' parameter."""
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        """The method performing the learning (e.g. in Q-Learning, update the table)"""
        raise NotImplementedError

    @abstractmethod
    def observe(self, *args):
        """Called at each observation.
        E.g. in args there can be the S,A,R,S' tuple for save it in a buffer
        that will be read in the "learn" method."""
        raise NotImplementedError

    def reset(self):
        """action performed at the end of each episode
        Subclasses should call this method.
        """
        self.episode += 1
        self.episode_iteration = 0

    def update(self):
        """action performed at the end of each iteration
        Subclasses should call this method.
        """
        self.episode_iteration += 1
        self.iteration += 1

