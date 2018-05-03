from abc import abstractmethod
from typing import Set

from flloat.semantics.pl import PLInterpretation
from pythomata.base.Simulator import DFASimulator, Simulator
from pythomata.base.Symbol import Symbol
from pythomata.base.utils import Sink

from rltg.logic.RewardAutomaton import RewardAutomaton

class RewardSimulator(Simulator):

    @abstractmethod
    def get_immediate_reward(self, q, q_prime):
        raise NotImplementedError

    @abstractmethod
    def is_failed(self) -> bool:
        raise NotImplementedError


class RewardAutomatonSimulator(DFASimulator, RewardSimulator):
    def __init__(self, dfa:RewardAutomaton):
        super().__init__(dfa)
        self.visited_states = {self.cur_state}

    def reset(self):
        super().reset()
        self.visited_states = {self.cur_state}

    def make_transition(self, s:Set[Symbol]):
        i = PLInterpretation(s)
        old_state = self.cur_state
        super().make_transition(i)
        reward = self.get_immediate_reward(old_state, self.cur_state)
        self.visited_states.add(self.cur_state)

        return reward

    def get_immediate_reward(self, q, q_prime):
        q_id = self.id2state[q]
        q_prime_id = self.id2state[q_prime]
        return self.dfa.get_immediate_reward(q_id, q_prime_id)

    def is_failed(self):
        return self.id2state[self.cur_state] in self.dfa.failure_states

    def get_cur_state(self):
        return self.cur_state
