from flloat.base.Alphabet import Alphabet
from flloat.base.Symbol import Symbol
from flloat.syntax.ldlf import LDLfFormula
from pythomata.base.DFA import DFA
from typing import Set


class RewardAutomaton(DFA):
    def __init__(self, dfa:DFA, alphabet:Alphabet, f:LDLfFormula, reward):
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

    @staticmethod
    def _fromFormula(alphabet:Set[Symbol], f:LDLfFormula, reward):
        dfa = f.to_automaton(alphabet, determinize=True, minimize=True)
        return RewardAutomaton(dfa, dfa.alphabet, f, reward)

    def get_reward(self):
        return self.reward

    def complete(self):
        return RewardAutomaton(self._dfa.complete(), self.alphabet, self.f, self.reward)

    def minimize(self):
        return RewardAutomaton(self._dfa.minimize(), self.alphabet, self.f, self.reward)

    def trim(self):
        return RewardAutomaton(self._dfa.trim(), self.alphabet, self.f, self.reward)
