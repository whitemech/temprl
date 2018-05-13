from flloat.flloat import DFAOTF
from typing import Set, List

from flloat.semantics.pl import PLInterpretation
from flloat.utils import powerset
from pythomata.base.Alphabet import Alphabet
from pythomata.base.DFA import DFA
from pythomata.base.Symbol import Symbol

from rltg.logic.PartialRewardAutomaton import PartialRewardAutomaton
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
        self.exploring_states = {}
        self.exploration_bonuses = {}

        dfa = DFA(alphabet, frozenset(self.states), self.initial_state, frozenset(self.final_states), self.transition_function)
        # keep the current and the last automaton in order to apply Dynamic Reward Shaping
        self._automaton = PartialRewardAutomaton(dfa, alphabet, dfaotf.f, reward, self.exploring_states, gamma=gamma)
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
        return self.state2id[new_state]


    def get_immediate_reward(self, q, q_prime, is_terminal_state=False):
        q_id = q
        q_prime_id = q_prime
        r = self.get_dynamic_reward(q_id, q_prime_id, is_terminal_state=is_terminal_state)
        # if r!=0.0: print(self.episode, self.it, str(self.dfaotf.f)[:9], q_id, q_prime_id, str(r))
        return r

    def get_dynamic_reward(self, q, q_prime, is_terminal_state=False):

        phi_ = self._automaton.potential_function
        phi_prime_ = self._automaton_prime.potential_function

        phi = lambda x: self.exploration_bonuses.get(q, 0) if q in self.exploring_states else 0 + phi_(q)
        phi_prime = lambda x: self.exploration_bonuses.get(q_prime, 0) if q_prime in self.exploring_states else 0 + phi_prime_(q_prime)

        if q_prime in self.failure_states:
            assert is_terminal_state
            # print("failed ", q, q_prime)
            # print("reward ", - phi(q))
            return - phi(q)
        elif q_prime in self._automaton_prime.accepting_states:
            assert is_terminal_state
            return self.reward - phi(q)
        elif is_terminal_state:
            return - phi(q)

        r = self.gamma * phi_prime(q_prime) - phi(q)
        if r<0.0:
            print(str(self.dfaotf.f)[:9], "NEGATIVE: ", q_prime, q, " -> ", self.gamma,"*",phi_prime(q_prime), "-",phi(q), "->", r)
        elif r > 0.0:
            print(str(self.dfaotf.f)[:9], "POSITIVE: ", q_prime, q, " -> ", self.gamma,"*",phi_prime(q_prime), "-", phi(q),"->", r)

        # if q in self.exploration_bonuses:
        #     if self.exploration_bonuses[q]<1e-3:
        #         self.exploration_bonuses[q] = 0
        #     self.exploration_bonuses[q] *= 0.99

        if q_prime in self.exploration_bonuses:
            if self.exploration_bonuses[q_prime]<1e-4*self.reward:
                self.exploration_bonuses[q_prime] = 0
            self.exploration_bonuses[q_prime] *= 0.99

        return r

    def _is_failed_state(self, state):
        return state == frozenset()

    def is_failed(self):
        return self._is_failed_state(self.dfaotf.cur_state)

    def _is_true_state(self, state):
        return state == frozenset()

    def is_true(self):
        return self.dfaotf.is_true()

    def word_acceptance(self, word: List[Symbol]):
        return self.dfaotf.word_acceptance([PLInterpretation(c) for c in word])

    def _update_from_transition(self, old_state, label, new_state):

        old_state_id, old_state_changed = self._add_state(old_state)
        new_state_id, new_state_changed = self._add_state(new_state)

        changed = old_state_changed or new_state_changed

        # TODO REMOVE DEBUG PRINT...
        # if changed: print("%s: changed new states"%self.episode, new_state not in self.state2id, old_state not in self.state2id, new_state_id,
        #       old_state_id)

        if old_state_id not in self.transition_function or label not in self.transition_function[old_state_id]:
            changed = True
            self.transition_function.setdefault(old_state_id, {})[label] = new_state_id

            # TODO REMOVE DEBUG PRINT...
            # print("%s: changed transition"%self.episode, old_state_id not in self.transition_function, label not in self.transition_function[old_state_id], old_state_id, label, new_state_id)

        if self.is_failed() and new_state_id not in self.failure_states:
            self.failure_states.add(new_state_id)
            changed = True

            # TODO REMOVE DEBUG PRINT...
            # print("%s: changed failed"%self.episode, new_state_id)
        elif self.is_true() and new_state_id not in self.final_states:
            self.final_states.add(new_state_id)
            changed = True

            # TODO REMOVE DEBUG PRINT...
            # print("%s: changed true "%self.episode, new_state_id)

        if changed:
            # print("changed")
            dfa = DFA(self.alphabet, frozenset(self.states), self.initial_state, frozenset(self.final_states),
                  self.transition_function)

            exploring_states = set()
            new_exploration_bonuses = {}
            for s in self._automaton.failure_states:
                if s == self.initial_state:
                    continue
                if not self._is_failed_state(self.id2state[s]):
                    exploring_states.add(s)
                    new_exploration_bonuses[s] = self.exploration_bonuses.get(s, self.reward)
                    # new_exploration_bonuses[s] = 0
            self.exploration_bonuses = new_exploration_bonuses
            if not self._is_failed_state(new_state) and new_state_id not in self.failure_states and new_state_id not in self.exploration_bonuses:
                exploring_states.add(new_state_id)
                self.exploration_bonuses[new_state_id] = self.reward
                # self.exploration_bonuses[new_state_id] = 0

            self.exploring_states = exploring_states
            self._automaton = self._automaton_prime
            self._automaton_prime = PartialRewardAutomaton(dfa, self.alphabet, self.dfaotf.f, self.reward, exploring_states, gamma=self.gamma)
            self._automaton_prime.to_dot("automata/%s/%s-%s" % (str(self.dfaotf.f)[:9],self.episode, self.it))
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
