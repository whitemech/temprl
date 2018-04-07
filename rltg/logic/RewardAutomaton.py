from pythogic.base.Alphabet import Alphabet
from pythogic.base.Formula import Formula
from pythogic.base.utils import _to_pythomata_dfa
from pythogic.ldlf_empty_traces.LDLf_EmptyTraces import LDLf_EmptyTraces
from pythomata.base.DFA import DFA


class RewardAutomaton(DFA):
    def __init__(self, dfa:DFA, alphabet:Alphabet, f:Formula, reward):
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
    def _fromFormula(alphabet:Alphabet, f:Formula, reward):
        ldlf = LDLf_EmptyTraces(alphabet)
        nfa_dict = ldlf.to_nfa(f)
        dfa = _to_pythomata_dfa(nfa_dict)
        return RewardAutomaton(dfa, dfa.alphabet, f, reward)

    def _fromDFA(self, dfa:DFA):
        return

    def get_reward(self):
        return self.reward

    def complete(self):
        return RewardAutomaton(self._dfa.complete(), self.alphabet, self.f, self.reward)
