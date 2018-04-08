from typing import List

from rltg.agents.RLAgent import RLAgent
from rltg.agents.feature_extraction import FeatureExtractor, RobotFeatureExtractor
from rltg.agents.brains.Brain import Brain
from rltg.agents.exploration_policies import ExplorationPolicy
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator


class TGAgent(RLAgent):
    """Temporal Goal agent"""

    def __init__(self,
                 sensors: RobotFeatureExtractor,
                 exploration_policy:ExplorationPolicy,
                 brain:Brain,
                 temporal_evaluators:List[TemporalEvaluator]):
        super().__init__(sensors, exploration_policy, brain)
        self.temporal_evaluators = temporal_evaluators

    # TODO: make this component more generic, maybe in a separate module
    def state_extractor(self, world_state, automata_states: List):
        # state = tuple([world_state]+list(automata_states))
        state = RobotFeatureExtractor.RobotState(world_state, tuple(automata_states))
        return state

    # TODO: make this component more generic, maybe in a separate module
    def reward_extractor(self, world_reward, automata_rewards: List):
        res = world_reward + sum(automata_rewards)
        return res

    def act(self, state):
        sw = state
        automata_states = [te.get_state() for te in self.temporal_evaluators]
        new_state = self.state_extractor(sw, automata_states)
        return super().act(new_state)

    def observe(self, state, action, reward, state2):
        old_states_automata = [te.get_state() for te in self.temporal_evaluators]
        states_automata, rewards_automata = zip(*[te.update(state2) for te in self.temporal_evaluators])

        # TODO: this must be properly defined depending from the context
        old_state  = self.state_extractor(state,  automata_states=old_states_automata)
        new_state2 = self.state_extractor(state2, automata_states=states_automata)
        new_reward = self.reward_extractor(reward, rewards_automata)

        super().observe(old_state, action, new_reward, new_state2)

    def reset(self):
        super().reset()
        for te in self.temporal_evaluators:
            te.reset()
