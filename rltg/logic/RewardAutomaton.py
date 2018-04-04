from pythogic.base.Alphabet import Alphabet
from pythogic.base.Formula import Formula
from pythogic.base.utils import _to_pythomata_dfa
from pythogic.ldlf_empty_traces.LDLf_EmptyTraces import LDLf_EmptyTraces
from pythomata.base.DFA import DFA


class RewardAutomaton(DFA):
    def __init__(self, alphabet:Alphabet, f:Formula, reward):
        dfa = self._fromFormula(alphabet, f)
        super().__init__(
            dfa.alphabet,
            dfa.states,
            dfa.initial_state,
            dfa.accepting_states,
            dfa.transition_function
        )
        self.reward = reward

    @staticmethod
    def _fromFormula(alphabet:Alphabet, f:Formula):
        ldlf = LDLf_EmptyTraces(alphabet)
        nfa_dict = ldlf.to_nfa(f)
        dfa = _to_pythomata_dfa(nfa_dict)
        return dfa

    def get_reward(self):
        return self.reward

