"""Base class for every RL agent."""

import _pickle as pickle

from rltg.agents.brains.Brain import Brain
from rltg.agents.exploration_policies import ExplorationPolicy
from rltg.agents.feature_extraction import RobotFeatureExtractor


class RLAgent(object):

    def __init__(self, sensors: RobotFeatureExtractor, exploration_policy:ExplorationPolicy, brain:Brain, eval:bool=False):
        """
        :param sensors:             feature extractor from the environment observation.
        :param exploration_policy:  A wrapper class to the exploration policy during the learning process.
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
        self.exploration_policy = exploration_policy
        self.brain = brain
        self.eval = eval

    def set_eval(self, eval:bool):
        """Setter method for "eval" field."""
        self.eval = eval

    def act(self, state, best_action=False):
        """Extract the features and call _act
        :param state:       the state from which the agent makes a move.
        :raises ValueError: if It the state is not contained into sensors.input_space
        :returns the chosen action
        """
        features = self.sensors(state)
        return self._act(features, best_action=best_action)


    def _act(self, features, best_action=False):
        """From features, choose an action.
        :param features: the state from which the agent makes a move.
        :returns the chosen action
        """
        action = None
        if not self.eval and not best_action:
            # if in evaluation mode, do not explore.
            action = self.exploration_policy.explore()
        if action is None:
            # if the exploration policy does not specify any action
            # or if we are either in evaluation mode or we need the optimal action,
            # then ask the action to the policy
            action = self.brain.choose_action(features, optimal=best_action)
        return action

    def observe(self, state, action, reward, state2):
        """Called at each observation. """
        features_1 = self.sensors(state)
        features_2 = self.sensors(state2)
        self._observe(features_1, action, reward, features_2)

    def _observe(self, features, action, reward, features2):
        self.brain.observe(features, action, reward, features2)

    def replay(self):
        if not self.eval:
            self.brain.learn()

    def update(self):
        """Called at the end of each iteration.
        It MUST be called only once."""
        if not self.eval:
            self.brain.update()
            self.exploration_policy.update()

    def reset(self):
        """Called at the end of each episode.
        It MUST be called only once."""
        if not self.eval:
            self.brain.reset()
            self.exploration_policy.reset()

    def save(self, filepath):
        with open(filepath + "/exploration_policy.dump", "wb") as fout:
            self.exploration_policy.reset()
            pickle.dump(self.exploration_policy, fout)
        with open(filepath + "/brain.dump", "wb") as fout:
            self.brain.reset()
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
