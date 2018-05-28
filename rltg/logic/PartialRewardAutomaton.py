import logging
from copy import copy
from itertools import chain
from typing import Any, Set, List

from flloat.base.Alphabet import Alphabet
from flloat.flloat import DFAOTF
from flloat.semantics.pl import PLInterpretation
from flloat.syntax.ldlf import LDLfFormula
from flloat.base.Symbol import Symbol
from pythomata.base.DFA import DFA

from rltg.logic.RewardAutomatonSimulator import RewardSimulator

from rltg.logic.RewardAutomaton import RewardAutomaton
from rltg.utils.misc import compute_levels, _potential_function, mydefaultdict


class PartialRewardAutomaton(RewardAutomaton, RewardSimulator):
    def __init__(self, alphabet:Alphabet, f:LDLfFormula, reward, gamma=0.99):

        self.dfaotf = DFAOTF(f)
        self.alphabet = alphabet
        self.f = f
        self.reward = reward

        self.state2id = {self.dfaotf.cur_state: 0}
        self.id2state = {0: self.dfaotf.cur_state}

        self.states = {0}
        self.initial_state = 0
        self.accepting_states = {0} if self.dfaotf.is_true() else set()

        self.transition_function = {}
        self.inverse_transition_function = {}
        self.links = mydefaultdict(set())

        self.failure_states = set() if self._is_failed_state(self.dfaotf.cur_state) else {0}

        self.gamma = gamma
        self.episode = 0
        self.it = 0

        self.potentials_prime = mydefaultdict(0)


    def dfs(self, dfa, state, visited, border_states):
        leaves = []
        visited.add(state)
        exists_next = False
        if state in self.accepting_states:
            return leaves
        for a in dfa.transition_function.get(state, {}):
            next_state = dfa.transition_function[state][a]
            if next_state not in visited and not self._is_failed_state(self.id2state[next_state]):
                exists_next = True
                leaves += self.dfs(dfa, next_state, visited, border_states)
        if not exists_next and not self._is_failed_state(self.id2state[state]) and state in border_states:
            leaves.append(state)
        return leaves


    def _compute_levels(self):
        temp_dfa = DFA(self.alphabet, self.states, self.initial_state, self.accepting_states, self.transition_function)
        state2level, max_level, failure_states = compute_levels(temp_dfa, temp_dfa.accepting_states)

        border_states = set(filter(lambda x: not self._is_failed_state(self.id2state[x]), failure_states))
        true_failure_state = set(filter(lambda x: self._is_failed_state(self.id2state[x]), failure_states))
        true_states = set(filter(lambda x: not self._is_failed_state(self.id2state[x]) and x not in border_states, temp_dfa.states))
        potentials = {s: _potential_function(s, self.initial_state, state2level, self.reward) for s in chain(true_failure_state, true_states)}

        # compute levels from initial state
        # level = 0
        # visited = set()
        # to_visit = set()
        # to_visit_next = {self.initial_state}
        # reachability_levels = {self.initial_state:0}
        #
        # changed = True
        # while changed:
        #     changed = False
        #     level += 1
        #     to_visit = to_visit_next
        #     to_visit_next = set()
        #
        #     for s in to_visit:
        #         if s in visited:
        #             continue
        #         else:
        #             visited.add(s)
        #         s_potential = potentials[s]
        #         for a in temp_dfa.transition_function.get(s, []):
        #             s_prime = temp_dfa.transition_function[s][a]
        #
        #             if s_prime not in reachability_levels:
        #                 reachability_levels[s_prime] = level
        #             elif level < reachability_levels[s_prime]:
        #                 reachability_levels[s_prime] = level
        #
        #             old_potential = potentials.get(s_prime, self.reward)
        #             # if potential is computed correctly
        #             if s_prime not in failure_states:
        #                 potentials[s_prime] = _potential_function(s, self.initial_state, state2level, self.reward)
        #             elif s_potential < old_potential:
        #                 potentials[s_prime] = s_potential
        #
        #             to_visit_next.add(s_prime)
        #             changed = True


        # least fixpoint with the new state just discovered
        leaves = self.dfs(temp_dfa, self.initial_state, set(), border_states)
        border_state2level, border_max_level, _ = compute_levels(temp_dfa, set(leaves))
        for b in border_states:
            potentials[b] = _potential_function(b, self.initial_state, border_state2level, self.reward)

        logging.debug("border states: {:15}, leaves: {:10}".format(str(border_states), str(leaves)))

        return potentials



    def potential_function(self, q, is_terminal_state=False, is_prime=False):
        if is_terminal_state:
            return 0
        elif is_prime:
            return self.potentials_prime[q]
        else:
            return self.potentials[q]


    def _update_from_transition(self, old_state, label, new_state):

        old_state_id, old_state_changed = self._add_state(old_state)
        new_state_id, new_state_changed = self._add_state(new_state)
        transition_already_exists = False
        is_failure_state = False
        is_final_state = False

        if old_state_id not in self.transition_function or label not in self.transition_function[old_state_id]:
            transition_already_exists = new_state_id in self.links[old_state_id]
            self.links[old_state_id].add(new_state_id)
            self.transition_function.setdefault(old_state_id, {})[label] = new_state_id

        if self._is_failed_state(new_state) and new_state_id not in self.failure_states:
            self.failure_states.add(new_state_id)
            is_failure_state = True

        elif self.dfaotf._is_true_state(new_state) and new_state_id not in self.accepting_states:
            self.accepting_states.add(new_state_id)
            is_final_state = True

        changed = old_state_changed or (new_state_changed and not is_failure_state and not transition_already_exists)

        if changed:

            self.potentials = self.potentials_prime
            self.potentials_prime = self._compute_levels()

            dfa = DFA(self.alphabet, frozenset(self.states), self.initial_state, frozenset(self.accepting_states), self.transition_function)
            dfa.to_dot("automata/%s/%s-%s" % (str(self.dfaotf.f)[:9],self.episode, self.it))

            logging.debug(
                "episode: {:6d}, step: {:5d}, update automaton on-the-fly, prev_state: {:3}, next_state_id: {:3}, label: {:10}, is fail: {:5}, is final: {:5}".format(
                    self.episode, self.it, old_state_id, new_state_id, str(label), str(new_state_id in self.failure_states), str(new_state_id in self.accepting_states)
                ))
            logging.debug("new potentials: %s", self.potentials_prime)
        else:
            self.potentials = copy(self.potentials_prime)

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

    def _is_failed_state(self, state):
        return state == frozenset()

    def is_failed(self):
        return self._is_failed_state(self.dfaotf.cur_state)

    def make_transition(self, s:Set[Symbol]) -> Any:
        old_state = self.dfaotf.get_current_state()
        i = PLInterpretation(s)
        self.dfaotf.make_transition(i)
        new_state = self.dfaotf.cur_state
        self._update_from_transition(old_state, i, new_state)
        self.it += 1
        return self.get_current_state()

    def is_true(self) -> bool:
        return self.dfaotf.is_true()

    def reset(self) -> Any:
        self.dfaotf.reset()
        self.episode += 1
        self.it = 0

    def get_current_state(self):
        return self.state2id[self.dfaotf.get_current_state()]

    def word_acceptance(self, word: List[Symbol]):
        raise NotImplementedError
