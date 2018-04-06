"""Abstract class for every RL agent."""
from abc import ABC

import _pickle as pickle

from rltg.agents.brains.Brain import Brain
from rltg.agents.exploration_policies import ExplorationPolicy
from rltg.agents.feature_extraction import FeatureExtractor


class RLAgent(ABC):

    def __init__(self, sensors: FeatureExtractor, exploration_policy:ExplorationPolicy, brain:Brain):
        if sensors.output_space != brain.observation_space:
            raise ValueError("space dimensions are not compatible.")
        self.sensors = sensors
        self.exploration_policy = exploration_policy
        self.brain = brain

    def act(self, state):
        action = self.exploration_policy.explore()
        if action is None:
            features = self.sensors(state)
            action = self.brain.choose_action(features)
        return action

    def observe(self, state, action, reward, state2):
        """Called at each observation"""
        features_1 = self.sensors(state)
        features_2 = self.sensors(state2)
        self.brain.observe(features_1, action, reward, features_2)

    def replay(self):
        self.brain.learn()

    def update(self):
        """Called at the end of each iteration"""
        self.brain.update()
        self.exploration_policy.update()

    def reset(self):
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
