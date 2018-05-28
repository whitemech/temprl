import pickle
from abc import ABC

from typing import Tuple

from rltg.agents.brains.Brain import Brain
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.utils.misc import AgentObservation
from rltg.utils.RewardShaping import RewardShaping


class Agent(ABC):
    def __init__(self, sensors: RobotFeatureExtractor, brain:Brain, reward_shapers:Tuple[RewardShaping]=(), eval:bool=False):
        """
        :param sensors:             feature extractor from the environment observation.
        :param brain:               The Brain, a wrapper for the actual learning algorithm.
        :param eval:                If eval is True, prevent the brain to learn (testing mode).
                                    otherwise work as usual (learning mode)
                                    default: False

        :raises ValueError          if the space of the feature extractor output is different from the space of
                                    the brain input.
        """
        if brain.observation_space and sensors.output_space != brain.observation_space:
            raise ValueError("space dimensions are not compatible.")
        self.sensors = sensors
        self.brain = brain
        self.reward_shapers = reward_shapers
        self.eval = eval

    def set_eval(self, eval:bool):
        """Setter method for "eval" field."""
        self.eval = eval
        self.brain.set_eval(eval)

    def observe(self, state, action, world_reward, state2, **kwargs):
        """Called at each observation. """
        features_1 = self.sensors(state)
        features_2 = self.sensors(state2)
        reward = world_reward + sum(rs(features_1, features_2) for rs in self.reward_shapers)
        return self._observe(features_1, action, reward, features_2, **kwargs)

    def _observe(self, features, action, reward, features2, **kwargs):
        obs = AgentObservation(features, action, reward, features2)
        self.brain.observe(obs)
        return obs

    def update(self):
        """Called at the end of each iteration.
        It MUST be called only once for iteration."""
        self.brain.update()

    def start(self, state):
        features = self.sensors(state)
        return self.brain.start(features)

    def step(self, obs: AgentObservation):
        return self.brain.step(obs)

    def end(self, obs: AgentObservation):
        """Called at the end of each episode.
        It MUST be called only once for episode."""
        self.brain.end(obs)

    def save(self, filepath):
        with open(filepath + "/obj.dump", "wb") as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(filepath):
        with open(filepath + "/obj.dump", "rb") as fin:
            agent = pickle.load(fin)
        return agent
