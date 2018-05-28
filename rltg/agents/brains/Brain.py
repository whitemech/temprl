from abc import ABC, abstractmethod

from gym.core import Space

from rltg.utils.misc import AgentObservation
from rltg.agents.policies.Policy import Policy


class Brain(ABC):
    """The class which implements the core of the algorithms"""

    def __init__(self, observation_space:Space, action_space:Space, policy:Policy):
        """
        :param observation_space: instance of Space or None. If None, it means that the observation space
                                  is not known a priori and so it is not needed for the algorithm.
        :param action_space:      instance of Space.
        """

        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = policy

        self.episode = 0
        self.iteration = 0
        self.episode_iteration = 0
        self.obs_history = []
        self.total_reward = 0

    def set_eval(self, eval:bool):
        self.eval = eval
        self.policy.set_eval(eval)

    @abstractmethod
    def choose_action(self, state, **kwargs):
        """From a state, return the action for the implemented approach.
        e.g. in Q-Learning, select the argmax of the Q-values relative to the 'state' parameter."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, obs:AgentObservation, *args, **kwargs):
        """Called at each observation.
        E.g. in args there can be the S,A,R,S' tuple for save it in a buffer
        that will be read in the "learn" method."""
        self.obs_history.append(obs)
        self.total_reward += obs.reward

    def update(self, *args, **kwargs):
        """action performed at the end of each iteration
        Subclass implementations should call this method.
        """
        self.episode_iteration += 1
        self.iteration += 1
        self.policy.update()

    def start(self, state):
        if not self.eval:
            self.episode += 1
        self.episode_iteration = 0
        self.total_reward = 0
        self.policy.reset()
        self.obs_history = []

    @abstractmethod
    def step(self, obs: AgentObservation, *args, **kwargs):
        """The method performing the learning (e.g. in Q-Learning, update the table)"""
        raise NotImplementedError

    def end(self, obs:AgentObservation, *args, **kwargs):
        """action performed at the end of each episode
        Subclasses implementations should call this method.
        """
        self.episode_iteration += 1


