from typing import List

import pickle

import os

import re

from rltg.agents.RLAgent import RLAgent
from rltg.agents.feature_extraction import FeatureExtractor, RobotFeatureExtractor, TupleFeatureExtractor
from rltg.agents.brains.Brain import Brain
from rltg.agents.exploration_policies import ExplorationPolicy
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from gym.spaces import Dict, Discrete, Box, Tuple


class TGAgent(RLAgent):
    """Temporal Goal agent"""

    def __init__(self,
                 sensors: RobotFeatureExtractor,
                 exploration_policy:ExplorationPolicy,
                 brain:Brain,
                 temporal_evaluators:List[TemporalEvaluator]):
        assert len(temporal_evaluators)>=1
        super().__init__(sensors, exploration_policy, brain)
        self.temporal_evaluators = temporal_evaluators

        # compute the feature space. It is the cartesian product between
        # the robot feature space output and the automata state space
        # sensors.output_space is expected to be a Tuple.
        robot_feature_space = sensors.output_space
        automata_state_spaces = [temp_eval.get_state_space() for temp_eval in temporal_evaluators]

        # total feature space = (robot feature space, automata 1 state space, automata 2 state space, ... )
        feature_space = Tuple(robot_feature_space.spaces + tuple(automata_state_spaces))

        # Check if the brain has the same input space dimensions,
        # but only if the brain has specified an observation space.
        if brain.observation_space and feature_space != brain.observation_space:
            raise ValueError("The brain has incompatible observation space: {} instead of {}"
                             .format(brain.observation_space, feature_space))

        # map every state from the space (N0, N1, ..., Nn) to a discrete space of dimension N0*N1*...*Nn-1
        self._from_tuple_to_int = TupleFeatureExtractor(feature_space)

    # TODO: allow customization of this component by modularization
    def state_extractor(self, world_state, automata_states: List):
        # the state is a tuple: (features, A1 state, ..., An state)
        state = self.sensors(world_state) + tuple(automata_states)
        collapsed_state = self._from_tuple_to_int(state)
        return collapsed_state

    # TODO: allow customization of this component by modularization
    def reward_extractor(self, world_reward, automata_rewards: List):
        res = world_reward + sum(automata_rewards)
        return res

    def act(self, state, **kwargs):
        sw = state
        automata_states = [te.get_state() for te in self.temporal_evaluators]
        features = self.state_extractor(sw, automata_states)
        return super()._act(features, **kwargs)

    def observe(self, state, action, reward, state2):
        # get the current automata states
        old_states_automata = [te.get_state() for te in self.temporal_evaluators]

        # update the automata states given the new observed state and collect the reward
        states_automata, rewards_automata = zip(*[te.update(action, state2) for te in self.temporal_evaluators])

        old_state  = self.state_extractor(state,  old_states_automata)
        new_state2 = self.state_extractor(state2, states_automata)
        new_reward = self.reward_extractor(reward, rewards_automata)
        super()._observe(old_state, action, new_reward, new_state2)

    def reset(self):
        super().reset()
        for te in self.temporal_evaluators:
            te.reset()

    def save(self, filepath):
        super().save(filepath)
        for idx, te in enumerate(self.temporal_evaluators):
            with open(filepath + "/te%02d.dump"%idx, "wb") as fout:
                te.reset()
                pickle.dump(te, fout)

    def load(self, filepath):
        super().load(filepath)
        temporal_evaluators_files = sorted([f for f in os.listdir(filepath) if re.match(r'te.*\.dump', f)])
        temporal_evaluators = []
        for idx, te_name in enumerate(temporal_evaluators_files):
            with open(filepath + "/%s" % te_name, "rb") as fin:
                te = pickle.load(fin)
                temporal_evaluators.append(te)
        self.temporal_evaluators = temporal_evaluators
