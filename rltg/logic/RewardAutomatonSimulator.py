from abc import abstractmethod
from typing import Set

from flloat.semantics.pl import PLInterpretation
from pythomata.base.Simulator import DFASimulator, Simulator
from pythomata.base.Symbol import Symbol
from pythomata.base.utils import Sink

from rltg.logic.CompleteRewardAutomaton import CompleteRewardAutomaton

class RewardSimulator(Simulator):

    def potential_function(self, q, is_terminal_state=False):
        raise NotImplementedError

    @abstractmethod
    def is_failed(self) -> bool:
        raise NotImplementedError


class RewardAutomatonSimulator(DFASimulator, RewardSimulator):
    def __init__(self, dfa:CompleteRewardAutomaton):
        super().__init__(dfa)
        self.visited_states = {self.cur_state}

    def reset(self):
        super().reset()
        self.visited_states = {self.cur_state}

    def make_transition(self, s:Set[Symbol]):
        i = PLInterpretation(s)
        super().make_transition(i)
        self.visited_states.add(self.cur_state)
        return self.cur_state

    def potential_function(self, q, is_terminal_state=False):
        return self.dfa.potential_function(q, is_terminal_state=is_terminal_state)

    def is_failed(self):
        return self.id2state[self.cur_state] in self.dfa.failure_states

    def get_current_state(self):
        return self.cur_state
