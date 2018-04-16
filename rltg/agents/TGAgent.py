from typing import List

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
        super().__init__(sensors, exploration_policy, brain)
        self.temporal_evaluators = temporal_evaluators

        # compute the feature space. It is the cartesian product between
        # the robot feature space output and the automata state space
        robot_feature_space = sensors.output_space
        automata_states_len = [len(temp_eval.simulator.state2id) for temp_eval in temporal_evaluators]

        # total feature space = (robot feature space, automata 1 state space, automata 2 state space, ... )
        feature_space = Tuple(robot_feature_space.spaces + tuple([Discrete(n) for n in automata_states_len]))

        # Check if the brain has the same input space dimensions,
        # but only if the brain has specified an observation space.
        if brain.observation_space and feature_space != brain.observation_space:
            raise ValueError("The brain has incompatible observation space: {} instead of {}"
                             .format(brain.observation_space, feature_space))

        # map every state from the space (N0, N1, ..., Nn) to a discrete space of dimension N0*N1*...*Nn-1
        self._from_tuple_to_int = TupleFeatureExtractor(feature_space)


    # TODO: make this component more generic, maybe in a separate module
    def state_extractor(self, world_state, automata_states: List):
        # the state is a tuple: (features, A1 state, ..., An state)
        state = self.sensors(world_state) + tuple(automata_states)
        collapsed_state = self._from_tuple_to_int(state)
        return collapsed_state

        # state = RobotFeatureExtractor._RobotState(world_state, tuple(automata_states))

    # TODO: make this component more generic, maybe in a separate module
    def reward_extractor(self, world_reward, automata_rewards: List):
        res = world_reward + sum(automata_rewards)
        return res

    def act(self, state):
        sw = state
        automata_states = [te.get_state() for te in self.temporal_evaluators]
        features = self.state_extractor(sw, automata_states)
        return super()._act(features)

    def observe(self, state, action, reward, state2):
        if self.eval:
            # evaluation mode: do nothing
            return

        # get the current automata states
        old_states_automata = [te.get_state() for te in self.temporal_evaluators]
        # update the automata states given the new observed state and collect the reward
        states_automata, rewards_automata = zip(*[te.update(state2) for te in self.temporal_evaluators])

        # TODO: this must be properly defined depending on the context
        old_state  = self.state_extractor(state,  old_states_automata)
        new_state2 = self.state_extractor(state2, states_automata)
        new_reward = self.reward_extractor(reward, rewards_automata)
        super()._observe(old_state, action, new_reward, new_state2)

    def reset(self):
        super().reset()
        for te in self.temporal_evaluators:
            te.reset()
