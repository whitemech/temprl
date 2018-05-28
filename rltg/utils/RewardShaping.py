from abc import abstractmethod

from rltg.logic.PartialRewardAutomaton import PartialRewardAutomaton
from rltg.logic.RewardAutomaton import RewardAutomaton


class RewardShaping(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, state, state2, *args, is_terminal_state=False, **kwargs):
        return self.shaping_reward(state, state2, is_terminal_state=is_terminal_state)

    @abstractmethod
    def potential_function(self, s, is_terminal_state=False):
        raise NotImplementedError

    def shaping_reward(self, state, state2, is_terminal_state=False):
        phi = self.potential_function
        reward = phi(state2, is_terminal_state=is_terminal_state) - phi(state)
        # if state!=state2:
        #     print(state, state2, phi(state), phi(state2, is_terminal_state=is_terminal_state), reward)
        return reward


class AutomatonRewardShaping(RewardShaping):
    def __init__(self, automaton:RewardAutomaton, *args, **kwargs):
        super().__init__()
        self.automaton = automaton

    def potential_function(self, s, is_terminal_state=False):
        return self.automaton.potential_function(s, is_terminal_state=is_terminal_state)

class DynamicAutomatonRewardShaping(RewardShaping):
    def __init__(self, automaton:PartialRewardAutomaton, *args, **kwargs):
        super().__init__()
        self.automaton = automaton

    def potential_function(self, s, is_terminal_state=False):
        raise NotImplementedError

    def shaping_reward(self, state, state2, is_terminal_state=False):
        phi = self.automaton.potential_function
        reward = phi(state2, is_prime=True, is_terminal_state=is_terminal_state) - phi(state)
        return reward
