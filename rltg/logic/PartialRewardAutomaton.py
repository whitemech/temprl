from copy import copy
from typing import Set

from flloat.base.Alphabet import Alphabet
from flloat.base.Symbol import Symbol
from flloat.syntax.ldlf import LDLfFormula
from pythomata.base.DFA import DFA
from rltg.logic.RewardAutomaton import RewardAutomaton

from rltg.logic.CompleteRewardAutomaton import CompleteRewardAutomaton


class PartialRewardAutomaton(RewardAutomaton):
    def __init__(self, dfa:DFA, alphabet:Alphabet, f:LDLfFormula, reward, exploring_states, gamma=0.99):
        super().__init__(
            dfa,
            alphabet,
            f,
            reward,
            gamma=gamma
        )

        self._dfa = dfa
        self.alphabet = alphabet
        self.f = f
        self.reward = reward

        self.reachability_levels, \
        self.max_level, \
        self.failure_states = self._compute_levels()

        self.gamma = gamma

        self.exploring_states = exploring_states
        self.exploring_state2min_potential_state = self._compute_partial_potentials()

    @staticmethod
    def _fromFormula(alphabet:Set[Symbol], f:LDLfFormula, reward, gamma=0.99):
        dfa = f.to_automaton(alphabet, determinize=True, minimize=True)
        return PartialRewardAutomaton(dfa, dfa.alphabet, f, reward, set(), gamma=gamma)

    def complete(self):
        return PartialRewardAutomaton(self._dfa.complete(), self.alphabet, self.f, self.reward, self.exploring_states, gamma=self.gamma)

    def minimize(self):
        return PartialRewardAutomaton(self._dfa.minimize(), self.alphabet, self.f, self.reward, self.exploring_states, gamma=self.gamma)

    def trim(self):
        return PartialRewardAutomaton(self._dfa.trim(), self.alphabet, self.f, self.reward, self.exploring_states, gamma=self.gamma)

    def potential_function(self, q):
        if q in self.exploring_state2min_potential_state:
            # print(str(self.f)[:7], "min potential ", q, self.exploring_state2min_potential_state[q])
            return super().potential_function(self.exploring_state2min_potential_state[q])
        else:
            return super().potential_function(q)

    def _compute_partial_potentials(self):
        exploring_state2min_potential_state = {}

        change = True

        to_be_visited = set()
        to_be_visited_next = set(filter(lambda x: x not in self.exploring_states,self.states))

        while change:
            change = False
            to_be_visited = to_be_visited_next
            to_be_visited_next = set()
            for s in to_be_visited:
                if s not in self.transition_function: continue
                cur_t = self.transition_function[s]
                for a in cur_t:
                    s_prime = cur_t[a]
                    if not s_prime in self.exploring_states:
                        continue

                    p = exploring_state2min_potential_state.get(s,super().potential_function(s))
                    # min potential seen
                    if s_prime not in exploring_state2min_potential_state or\
                        exploring_state2min_potential_state[s_prime]>p:
                        exploring_state2min_potential_state[s_prime] = exploring_state2min_potential_state.get(s, s)

                        to_be_visited_next.add(s_prime)
                        change = True

        return exploring_state2min_potential_state

    def _compute_levels(self):
        level = 0
        state2level = {final_state: level for final_state in self._dfa.accepting_states}

        z_current = set()
        z_next = set(self._dfa.accepting_states)
        while z_current != z_next:
            level += 1
            z_current = z_next
            z_next = copy(z_current)
            for s in self._dfa.states:
                if s in z_current:
                    continue
                for a in self.transition_function.get(s, []):
                    next_state = self.transition_function[s][a]
                    if next_state in z_current:
                        z_next.add(s)
                        state2level[s] = level

        z_current = z_next

        max_level = level - 1

        # levels for failure state (i.e. that cannot reach a final state)
        failure_states = set()
        for s in filter(lambda x: x not in z_current, self._dfa.states):
            state2level[s] = level
            failure_states.add(s)

        return state2level, max_level, failure_states
