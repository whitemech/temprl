from copy import copy
from flloat.base.Alphabet import Alphabet
from flloat.base.Symbol import Symbol
from flloat.syntax.ldlf import LDLfFormula
from pythomata.base.DFA import DFA
from typing import Set

from rltg.logic.RewardAutomaton import RewardAutomaton
from rltg.utils.misc import compute_levels, _potential_function


class CompleteRewardAutomaton(DFA, RewardAutomaton):
    def __init__(self, dfa:DFA, alphabet:Alphabet, f:LDLfFormula, reward, gamma=0.99):
        super().__init__(
            dfa.alphabet,
            dfa.states,
            dfa.initial_state,
            dfa.accepting_states,
            dfa.transition_function
        )
        self._dfa = dfa
        self.alphabet = alphabet
        self.f = f
        self.reward = reward

        self.reachability_levels,   \
        self.max_level,             \
        self.failure_states = self._compute_levels()

        self.gamma = gamma

    @staticmethod
    def _fromFormula(alphabet:Set[Symbol], f:LDLfFormula, reward, gamma=0.99):
        dfa = f.to_automaton(alphabet, determinize=True, minimize=True)
        return CompleteRewardAutomaton(dfa, dfa.alphabet, f, reward, gamma=gamma)

    def get_formula_reward(self):
        return self.reward

    def complete(self):
        return CompleteRewardAutomaton(self._dfa.complete(), self.alphabet, self.f, self.reward, gamma=self.gamma)

    def minimize(self):
        return CompleteRewardAutomaton(self._dfa.minimize(), self.alphabet, self.f, self.reward, gamma=self.gamma)

    def trim(self):
        return CompleteRewardAutomaton(self._dfa.trim(), self.alphabet, self.f, self.reward, gamma=self.gamma)

    def potential_function(self, q, is_terminal_state=False):
        return _potential_function(q, self._dfa.initial_state, self.reachability_levels, self.reward,
                                   is_terminal_state=is_terminal_state)

    def _compute_levels(self):
        return compute_levels(self._dfa, self._dfa.accepting_states)
