from copy import copy
from flloat.base.Alphabet import Alphabet
from flloat.base.Symbol import Symbol
from flloat.syntax.ldlf import LDLfFormula
from pythomata.base.DFA import DFA
from typing import Set


class RewardAutomaton(DFA):
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
        return RewardAutomaton(dfa, dfa.alphabet, f, reward, gamma=gamma)

    def get_formula_reward(self):
        return self.reward

    def get_immediate_reward(self, q, q_prime):
        phi = self.potential_function
        if q_prime in self.failure_states:
            return -self.reward/self.max_level
            # return 0


        r = self.gamma * phi(q_prime) - phi(q)
        if q != q_prime:
            # implementation trick: do not scale reward if we are in the same state
            r /= self.max_level
            r *= self.reward

        return r

    def complete(self):
        return RewardAutomaton(self._dfa.complete(), self.alphabet, self.f, self.reward,gamma=self.gamma)

    def minimize(self):
        return RewardAutomaton(self._dfa.minimize(), self.alphabet, self.f, self.reward, gamma=self.gamma)

    def trim(self):
        return RewardAutomaton(self._dfa.trim(), self.alphabet, self.f, self.reward,gamma=self.gamma)

    def potential_function(self, q):
        return self.max_level - self.reachability_levels[q]

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

        # levels for failure state (i.e. that cannot reach a final state)
        failure_states = set()
        for s in filter(lambda x: x not in z_current, self._dfa.states):
            state2level[s] = level
            failure_states.add(s)

        max_level = level - 1
        if max_level==0: max_level = 1
        return state2level, max_level, failure_states




"""
        The potential function works as the following:
        - Compute the level of reachability for the final state (precomputed):
        >>> level = self.reachability_levels[q]
        The lower, the nearer to any final state (zero means that q is a final state)
        - Make it "the higher, the nearer":
        >>> true_level = self.max_level - level
        - Normalize over the maximum level possible so that
        the nearer to any final state, the higher the ratio (from 0.0 to 1.0)
        >>> level_ratio = true_level / self.max_level
        - Give the partial reward simply by using the ratio just computed;
        >>> state_reward = level_ratio * self.reward
        :param q: the state of the automaton
        :return: the value of the potential function
        """
