from typing import Set

from flloat.semantics.pl import PLInterpretation
from pythomata.base.Simulator import DFASimulator
from pythomata.base.Symbol import Symbol
from pythomata.base.utils import Sink

from rltg.logic.RewardAutomaton import RewardAutomaton


class RewardAutomatonSimulator(DFASimulator):
    def __init__(self, dfa:RewardAutomaton):
        super().__init__(dfa)
        self.visited_states = {self.cur_state}

    def reset(self):
        super().reset()
        self.visited_states = {self.cur_state}

    def make_transition(self, s:Set[Symbol]):
        i = PLInterpretation(s)
        super().make_transition(i)
        reward = 0

        # get the reward only if it is the first time we visit the state
        if self.cur_state not in self.visited_states:
            # get the total reward if we are in the final state
            if self.is_true():
                reward = self.dfa.get_reward()
            else:
                # get a fraction of the total reward to facilitate exploration
                # if we are in the Sink state, fail (highly negative reward).
                # TODO: backpropagation of the reward
                if isinstance(self.id2state[self.cur_state], Sink):
                    reward = -10000
                else:
                    reward = self.dfa.get_reward()/10

            self.visited_states.add(self.cur_state)

        return reward


