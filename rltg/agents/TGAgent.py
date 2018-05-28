import logging
import os
import pickle
import re
from typing import List

from gym.spaces import Tuple

from rltg.agents.Agent import Agent
from rltg.agents.brains.Brain import Brain
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.utils.RewardShaping import AutomatonRewardShaping, DynamicAutomatonRewardShaping


class TGAgent(Agent):
    """Temporal Goal agent"""

    def __init__(self,
                 sensors: RobotFeatureExtractor,
                 brain:Brain,
                 temporal_evaluators: List[TemporalEvaluator],
                 reward_shaping: bool = True):
        assert len(temporal_evaluators)>=1

        if reward_shaping:
            reward_shapers = tuple([AutomatonRewardShaping(te.simulator) if not te.on_the_fly else
                                    DynamicAutomatonRewardShaping(te.simulator) for te in temporal_evaluators])
        else:
            reward_shapers = tuple()
        super().__init__(sensors, brain, reward_shapers=reward_shapers)
        self.temporal_evaluators = temporal_evaluators

        # compute the feature space. It is the cartesian product between
        # the robot feature space output and the automata state space
        # sensors.output_space is expected to be a Tuple.
        robot_feature_space = sensors.output_space
        automata_state_spaces = [temp_eval.get_state_space() for temp_eval in temporal_evaluators]

        # total feature space = (robot feature space, automata 1 state space, automata 2 state space, ... )
        feature_space = Tuple(tuple([robot_feature_space.spaces] + automata_state_spaces))

        # Check if the brain has the same input space dimensions,
        # but only if the brain has specified an observation space.
        if brain.observation_space and feature_space != brain.observation_space:
            raise ValueError("The brain has incompatible observation space: {} instead of {}"
                             .format(brain.observation_space, feature_space))

        # map every state from the space (N0, N1, ..., Nn) to a discrete space of dimension N0*N1*...*Nn-1
        # self._from_tuple_to_int = TupleFeatureExtractor(feature_space)

    def sync(self, action, state2):
        new_automata_state = tuple(temp_eval.update(action, state2) for temp_eval in self.temporal_evaluators)
        return new_automata_state

    def start(self, state):
        for t in self.temporal_evaluators:
            t.reset()
        automata_states = [temp_eval.get_state() for temp_eval in self.temporal_evaluators]
        transformed_state = self.state_extractor(state, automata_states)
        # return super().start(transformed_state)
        return self.brain.start(transformed_state)

    def state_extractor(self, world_state, automata_states: List):
        # the state is a tuple: (features, A_1 state, ..., A_n state)
        # A_i is the i_th automaton associated to the i_th temporal goal
        state = tuple([self.sensors(world_state)] + list(automata_states))
        return state

    def reward_extractor(self, world_reward, automata_rewards: List):
        res = world_reward + sum(automata_rewards)
        return res

    def get_automata_state(self):
        return tuple(temp_eval.get_state() for temp_eval in self.temporal_evaluators)

    def observe(self, state, action, reward, state2,
                old_automata_state=(), new_automata_state=(), is_terminal_state=False):

        automata_rewards = [te.reward if is_terminal_state and te.is_true() else 0 for te in self.temporal_evaluators]

        # apply reward shaping to the observed automata rewards
        for i in range(len(self.reward_shapers)):
            plus = self.reward_shapers[i](old_automata_state[i], new_automata_state[i], is_terminal_state=is_terminal_state)
            automata_rewards[i] += plus

        old_state  = self.state_extractor(state,  old_automata_state)
        new_state2 = self.state_extractor(state2, new_automata_state)
        new_reward = self.reward_extractor(reward, automata_rewards)

        if old_automata_state!=new_automata_state or is_terminal_state:
            logging.debug("episode: {:6d}, step: {:5d}, old automata states: {:10}, new automata state: {:10}, new reward: {:8.3f}".format(
                self.brain.episode, self.brain.episode_iteration, str(old_automata_state), str(new_automata_state), new_reward
            ))

        return super()._observe(old_state, action, new_reward, new_state2)

    def save(self, filepath):
        super().save(filepath)
        for idx, te in enumerate(self.temporal_evaluators):
            with open(filepath + "/te%02d.dump"%idx, "wb") as fout:
                te.reset()
                pickle.dump(te, fout)

    @staticmethod
    def load(filepath):
        rl_agent = Agent.load(filepath)
        temporal_evaluators_files = sorted([f for f in os.listdir(filepath) if re.match(r'te.*\.dump', f)])
        temporal_evaluators = []
        for idx, te_name in enumerate(temporal_evaluators_files):
            with open(filepath + "/%s" % te_name, "rb") as fin:
                te = pickle.load(fin)
                temporal_evaluators.append(te)
        temporal_evaluators = temporal_evaluators

        sensors = rl_agent.sensors
        brain = rl_agent.brain

        assert brain and sensors and temporal_evaluators is not None
        return TGAgent(sensors, brain, temporal_evaluators)
