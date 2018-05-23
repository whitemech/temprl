from copy import copy
from typing import Any, Set, List

from flloat.base.Alphabet import Alphabet
from flloat.flloat import DFAOTF
from flloat.semantics.pl import PLInterpretation
from flloat.syntax.ldlf import LDLfFormula
from flloat.base.Symbol import Symbol
from pythomata.base.DFA import DFA

from rltg.logic.RewardAutomatonSimulator import RewardSimulator

from rltg.logic.RewardAutomaton import RewardAutomaton


class PartialRewardAutomaton(RewardAutomaton, RewardSimulator):
    def __init__(self, alphabet:Alphabet, f:LDLfFormula, reward, bonuses={}, gamma=0.99):

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

        self.failure_states = set() if self._is_failed_state(self.dfaotf.cur_state) else {0}

        self.bonuses = bonuses
        self.potentials =       {0: self.reward if self.dfaotf._is_true_state(self.dfaotf.cur_state) else 0}
        self.potentials_prime = copy(self.potentials)

        self.reachability_levels, \
        self.max_level, \
        self.potentials = self._compute_levels()

        self.gamma = gamma
        self.episode = 0
        self.it = 0

    def get_immediate_reward(self, q, q_prime, is_terminal_state=False, reward_shaping=True):
        phi = lambda x: self.potentials[x]
        phi_prime = lambda x: self.potentials_prime[x] if not is_terminal_state else 0
        r = self.gamma * phi_prime(q_prime) - phi(q)
        if q_prime in self.accepting_states and is_terminal_state:
            r += self.reward
        if r!=0.0:
            print(r, phi_prime(q_prime), phi(q))

        return r


    def potential_function(self, q, is_terminal_state=False):
        pass
        # if q in self.exploring_state2min_potential_state:
        #     # print(str(self.f)[:7], "min potential ", q, self.exploring_state2min_potential_state[q])
        #     return super().potential_function(self.exploring_state2min_potential_state[q])
        # else:
        #     return super().potential_function(q)

    def _compute_levels(self):
        level = 0
        state2level = {final_state: level for final_state in self.accepting_states}
        potentials = {}

        z_current = set()
        z_next = set(self.accepting_states)
        while z_current != z_next:
            level += 1
            z_current = z_next
            z_next = copy(z_current)
            for s in self.states:
                if s in z_current:
                    continue
                for a in self.transition_function.get(s, []):
                    next_state = self.transition_function[s][a]
                    if next_state in z_current:
                        z_next.add(s)
                        state2level[s] = level
                        # overwritten after the whole computation
                        # potentials[s] = level

        z_current = z_next

        max_level = level - 1
        for s in state2level:
            if max_level == 0:
                potentials[s] = self.reward if s in self.accepting_states else 0
            else:
                potentials[s] = (max_level - state2level[s])/max_level * self.reward
                print("compute levels ", s, state2level[s], max_level, (max_level - state2level[s])/max_level)

        visited = set()
        to_be_visited = set()
        to_be_visited_next = {self.initial_state}
        change = True
        on_the_fly_potentials = {self.initial_state: potentials[self.initial_state] if self.initial_state in potentials
                                 else self.reward if self.dfaotf._is_true_state(self.id2state[self.initial_state]) else 0}


        while change:
            change = False
            to_be_visited = to_be_visited_next
            to_be_visited_next = set()
            for s in to_be_visited:
                if s in visited: continue
                visited.add(s)
                if s not in self.transition_function: continue
                cur_t = self.transition_function[s]
                for a in cur_t:
                    s_prime = cur_t[a]
                    to_be_visited_next.add(s_prime)
                    change = True

                    if s_prime not in self.failure_states and \
                        self._is_failed_state(self.id2state[s_prime]): self.failure_states.add(s_prime)

                    if s_prime in potentials:
                        # when a final state is found, every state that reaches it makes true this condition
                        # correct potential computed
                        on_the_fly_potentials[s_prime] = potentials[s_prime]
                    elif s_prime not in on_the_fly_potentials:
                        # first time we see this state. Set the potential to the potential of the previous state
                        if s in self.accepting_states:
                            on_the_fly_potentials[s_prime] = 0
                        else:
                            on_the_fly_potentials[s_prime] = on_the_fly_potentials[s] + self.bonuses.get(s_prime, 0) \
                                if not s_prime in self.failure_states else 0
                    elif s_prime not in self.failure_states and s not in self.accepting_states:
                        # we already seen this state
                        # set if the potential of the previous state is less than the current potential
                        # previous potential
                        p = on_the_fly_potentials[s_prime]
                        # current potential (can be also a "true potential" state)
                        cur_pot = on_the_fly_potentials[s] + self.bonuses.get(s_prime, 0)
                        on_the_fly_potentials[s_prime] = cur_pot if cur_pot < p else p
                        print(str(self.f)[:9 ], "on_the_fly_potentials ", on_the_fly_potentials)

                    if on_the_fly_potentials[s_prime] > self.reward:
                        on_the_fly_potentials[s_prime] = self.reward

        # levels for failure state (i.e. that cannot reach a final state)

        # failure_states = set()
        # for s in filter(lambda x: x not in on_the_fly_potentials, self.states):
        #     print("failure state: ",s)
        #     state2level[s] = level
        #     failure_states.add(s)
        #     # potentials[s] = 0
        #     on_the_fly_potentials[s] = 0

        # print("Update failure state potential")
        # for k in self.failure_states:
        #     on_the_fly_potentials[k] = 0

        return state2level, max_level, on_the_fly_potentials

    def _update_from_transition(self, old_state, label, new_state):

        # for b in self.bonuses:
        #     if self.bonuses[b]>0.0:
        #         # self.bonuses[b]*=0.9999
        #         self.bonuses[b]*=0
        #     if self.bonuses[b]<self.reward*1e-4:
        #         self.bonuses[b] = 0.0


        old_state_id, old_state_changed = self._add_state(old_state)
        new_state_id, new_state_changed = self._add_state(new_state)

        changed = old_state_changed or new_state_changed

        # TODO REMOVE DEBUG PRINT...
        if changed: print("%s: changed new states"%self.episode, new_state not in self.state2id, old_state not in self.state2id, new_state_id,
              old_state_id)

        if old_state_id not in self.transition_function or label not in self.transition_function[old_state_id]:
            changed = True
            self.transition_function.setdefault(old_state_id, {})[label] = new_state_id

            # TODO REMOVE DEBUG PRINT...
            print("%s: changed transition"%self.episode, old_state_id not in self.transition_function, label not in self.transition_function[old_state_id], old_state_id, label, new_state_id)

        if self._is_failed_state(new_state) and new_state_id not in self.failure_states:
            self.failure_states.add(new_state_id)
            changed = True

            # TODO REMOVE DEBUG PRINT...
            print("%s: changed failed"%self.episode, new_state_id)

        elif self.dfaotf._is_true_state(new_state) and new_state_id not in self.accepting_states:
            self.accepting_states.add(new_state_id)
            changed = True

            # TODO REMOVE DEBUG PRINT...
            print("%s: changed true "%self.episode, new_state_id)

        if new_state_changed:
            # self.bonuses[new_state_id] = self.reward if new_state_id not in self.failure_states else 0
            self.bonuses[new_state_id] = 0

        if changed:
            print("changed")
            self.potentials = copy(self.potentials_prime)
            self.reachability_levels, \
            self.max_level, \
            self.potentials_prime = self._compute_levels()

            print(str(self.f)[:9], " potentials: ",       self.potentials)
            print(str(self.f)[:9], " potentials prime: ", self.potentials_prime)
            print(str(self.f)[:9], " bonuses: ", self.bonuses)
            dfa = DFA(self.alphabet, frozenset(self.states), self.initial_state, frozenset(self.accepting_states), self.transition_function)
            dfa.to_dot("automata/%s/%s-%s" % (str(self.dfaotf.f)[:9],self.episode, self.it))
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

        for b in self.bonuses:
            if self.bonuses[b]>0.0:
                self.bonuses[b]*=0.99
                # self.bonuses[b]*=0
            if self.bonuses[b]<self.reward*1e-4:
                self.bonuses[b] = 0.0

    def get_current_state(self):
        return self.state2id[self.dfaotf.get_current_state()]

    def word_acceptance(self, word: List[Symbol]):
        raise NotImplementedError
