from flloat.flloat import DFAOTF
from typing import Set, List

from flloat.semantics.pl import PLInterpretation
from flloat.utils import powerset
from pythomata.base.Alphabet import Alphabet
from pythomata.base.DFA import DFA
from pythomata.base.Symbol import Symbol

from rltg.logic.RewardAutomaton import RewardAutomaton
from rltg.logic.RewardAutomatonSimulator import RewardSimulator


class PartialAutomatonSimulator(RewardSimulator):

    def __init__(self, dfaotf:DFAOTF, alphabet:Alphabet, reward, gamma=0.99):
        self.episode = 0
        self.it = 0

        self.dfaotf   = dfaotf
        self.alphabet = Alphabet({PLInterpretation(set(sym)) for sym in powerset(alphabet.symbols)})
        self.reward   = reward
        self.gamma    = gamma
        self.dfaotf.reset()
        initial_state = self.dfaotf.cur_state

        self.id2state = {0: initial_state}
        self.state2id = {initial_state: 0}

        self.states = {0}
        self.initial_state = 0
        self.transition_function = {}
        self.final_states = set()
        self.failure_states = set()

        # a list of (old_state, label, new_state) to keep track of the sequence of states and labels
        self.trace = []
        self.changed = False

        dfa = DFA(alphabet, frozenset(self.states), self.initial_state, frozenset(self.final_states), self.transition_function)
        # keep the current and the last automaton in order to apply Dynamic Reward Shaping
        self._automaton = RewardAutomaton(dfa, alphabet, dfaotf.f, reward, gamma=gamma)
        self._automaton_prime = self._automaton

        self.visited_states = {self.initial_state}

    def reset(self):
        self.dfaotf.reset()
        self.visited_states = {self.initial_state}

        self.episode += 1
        self.it = 0
        # (old state, label, new state) triple
        # for o, i, n in self.trace:
        #     self._update_from_transition(o, i, n)
        # self.trace = []

    def make_transition(self, s:Set[Symbol]):
        self.it+=1
        i = PLInterpretation(s)
        old_state = self.dfaotf.cur_state
        self.dfaotf.make_transition(i)

        new_state = self.dfaotf.cur_state
        self._update_from_transition(old_state, i, new_state)

        if len(self.final_states) > 0:
            # if the goal state has been discovered: least-fixpoint reward works!
            # this is to manage cases when
            if self.state2id[new_state] in self._automaton_prime.failure_states and not self.is_failed():
                return self.reward/10
            else:
                return self.get_immediate_reward(self.state2id[old_state], self.state2id[new_state])
        else:
            return self.get_partial_reward(old_state, new_state)


    def get_partial_reward(self, old_state, new_state):
        if self.is_true():
            return self.reward
        if self.is_failed():
            return -self.reward/10
        if not self.is_failed() and old_state != new_state and new_state not in self.visited_states:
            # give an optimistic reward to help exploration
            return self.reward/10
        return 0

    def get_immediate_reward(self, q, q_prime):
        phi = self._automaton.potential_function
        phi_prime = self._automaton_prime.potential_function

        r = self.gamma * phi_prime(q_prime) - phi(q)
        if q != q_prime:
            # implementation trick: do not scale reward if we are in the same state
            r /= self._automaton_prime.max_level
            r *= self.reward
        if r>11000:
            print(q, q_prime, phi(q), phi_prime(q_prime), r)

        return r

    def is_failed(self):
        return self.dfaotf.cur_state == frozenset()

    def is_true(self):
        return self.dfaotf.is_true()

    def word_acceptance(self, word: List[Symbol]):
        return self.dfaotf.word_acceptance([PLInterpretation(c) for c in word])

    def _update_from_transition(self, old_state, label, new_state):

        old_state_id, old_state_changed = self._add_state(old_state)
        new_state_id, new_state_changed = self._add_state(new_state)

        changed = old_state_changed or new_state_changed

        if changed: print("%s: changed new states"%self.episode, new_state not in self.state2id, old_state not in self.state2id, new_state_id,
              old_state_id)


        if old_state_id not in self.transition_function or label not in self.transition_function[old_state_id]:
            changed = True
            self.transition_function.setdefault(old_state_id, {})[label] = new_state_id
            print("%s: changed transition"%self.episode, old_state_id not in self.transition_function, label not in self.transition_function[old_state_id], old_state_id, label, new_state_id)

        if self.is_failed() and new_state_id not in self.failure_states:
            self.failure_states.add(new_state_id)
            changed = True
            print("%s: changed failed"%self.episode, new_state_id)
        elif self.is_true() and new_state_id not in self.final_states:
            self.final_states.add(new_state_id)
            changed = True
            print("%s: changed true "%self.episode, new_state_id)

        if changed:
            print("why")
            dfa = DFA(self.alphabet, frozenset(self.states), self.initial_state, frozenset(self.final_states),
                  self.transition_function)

            self._automaton = self._automaton_prime
            self._automaton_prime = RewardAutomaton(dfa, self.alphabet, self.dfaotf.f, self.reward, gamma=self.gamma)
            self._automaton_prime.to_dot("automata/%s-%s" % (self.episode, self.it))
        else:
            self._automaton = self._automaton_prime





    def _add_state(self, new_state):
        changed = False
        new_state_id = self.state2id.get(new_state, None)
        if new_state_id is None:
            # first time we see the state
            changed = True
            new_state_id = len(self.states)
            self.states.add(new_state_id)
            self.id2state[new_state_id] = new_state
            self.state2id[new_state] = new_state_id

        return new_state_id, changed

    def get_current_state(self):
        return self.state2id[self.dfaotf.cur_state]
