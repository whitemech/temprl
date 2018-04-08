"""Abstract class for every RL agent."""
from abc import ABC

import _pickle as pickle

from rltg.agents.brains.Brain import Brain
from rltg.agents.exploration_policies import ExplorationPolicy
from rltg.agents.feature_extraction import FeatureExtractor, RobotFeatureExtractor


class RLAgent(ABC):

    def __init__(self, sensors: RobotFeatureExtractor, exploration_policy:ExplorationPolicy, brain:Brain, eval:bool=False):
        if sensors.output_space != brain.observation_space:
            raise ValueError("space dimensions are not compatible.")
        self.sensors = sensors
        self.exploration_policy = exploration_policy
        self.brain = brain
        self.eval = eval

    def set_eval(self, eval:bool):
        """Setter method for "eval" field.
        If eval is True, prevent the brain to learn (testing mode)
        else work as usual (learning mode)"""
        self.eval = eval

    def act(self, state):
        """From the state, choose the action.
        :param state: the state from which the agent makes a move.
        :raises ValueError: if It the state is not contained into sensors.input_space"""
        action = None
        if not self.eval:
            action = self.exploration_policy.explore()
        if action is None:
            features = self.sensors(state)
            action = self.brain.choose_action(features)
        return action

    def observe(self, state, action, reward, state2):
        """Called at each observation. """
        features_1 = self.sensors(state)
        features_2 = self.sensors(state2)
        if not self.eval:
            self.brain.observe(features_1, action, reward, features_2)

    def replay(self):
        if not self.eval:
            self.brain.learn()

    def update(self):
        """Called at the end of each iteration"""
        if not self.eval:
            self.brain.update()
        self.exploration_policy.update()

    def reset(self):
        if not self.eval:
            self.brain.reset()
        self.exploration_policy.reset()

    def save(self, filepath):
        with open(filepath + "/exploration_policy.dump", "wb") as fout:
            pickle.dump(self.exploration_policy, fout)
        with open(filepath + "/brain.dump", "wb") as fout:
            pickle.dump(self.brain, fout)
        with open(filepath + "/sensors.dump", "wb") as fout:
            pickle.dump(self.sensors, fout)

    def load(self, filepath):
        with open(filepath + "/exploration_policy.dump", "rb") as fin:
            self.exploration_policy = pickle.load(fin)
        with open(filepath + "/brain.dump", "rb") as fin:
            self.brain = pickle.load(fin)
        with open(filepath + "/sensors.dump", "rb") as fin:
            self.sensors = pickle.load(fin)
