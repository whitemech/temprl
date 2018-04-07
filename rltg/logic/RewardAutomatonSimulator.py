from typing import List

from pythomata.base.DFA import DFA
from pythomata.base.Simulator import Simulator
from pythomata.base.Symbol import Symbol
from pythomata.base.utils import Sink

from rltg.logic.RewardAutomaton import RewardAutomaton


class RewardAutomatonSimulator(Simulator):
    def __init__(self, dfa:RewardAutomaton):
        super().__init__(dfa)
        self.visited_states = {self.cur_state}

    def reset(self):
        super().reset()
        self.visited_states = {self.cur_state}

    def make_transition(self, s:Symbol):
        super().make_transition(s)
        reward = 0
        if self.cur_state not in self.visited_states:
            if self.is_true():
                reward = self.dfa.get_reward()
            else:
                # TODO: backpropagation of the reward
                if isinstance(self.id2state[self.cur_state], Sink):
                    reward = -10000
                else:
                    reward = self.dfa.get_reward()/10

            self.visited_states.add(self.cur_state)

        return reward


