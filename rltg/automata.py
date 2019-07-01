from abc import abstractmethod
from copy import copy
from typing import Union, Set, List

from flloat.ldlf import LDLfFormula
from flloat.ltlf import LTLfFormula
from pythomata.base import Symbol
from pythomata.dfa import DFA
from pythomata.simulator import DFASimulator, Simulator

TemporalLogicFormula = Union[LTLfFormula, LDLfFormula]


class RewardAutomaton:

    @abstractmethod
    def potential_function(self, q, is_terminal_state=False):
        raise NotImplementedError


class RewardSimulator(Simulator):

    def potential_function(self, q, is_terminal_state=False):
        raise NotImplementedError

    @abstractmethod
    def is_failed(self) -> bool:
        raise NotImplementedError


class RewardDFA(DFA, RewardAutomaton):

    def __init__(self, dfa: DFA, reward):
        # TODO fix pythomata.DFA private attributes
        dfa = dfa.complete()
        super().__init__(
            set(dfa._states),
            set(dfa._alphabet),
            dfa._initial_state,
            set(dfa._accepting_states),
            dfa._transition_function
        )

        self.reward = reward

        self.reachability_levels,   \
        self.max_level,             \
        self.failure_states = self._compute_levels()

    @classmethod
    def from_formula(cls, f: TemporalLogicFormula, reward, alphabet: List[Symbol] = None):
        dfa = f.to_automaton(alphabet)
        return RewardDFA(dfa, reward)

    def potential_function(self, q, is_terminal_state=False):
        return _potential_function(q, self._initial_state, self.reachability_levels, self.reward,
                                   is_terminal_state=is_terminal_state)

    def _compute_levels(self):
        return _compute_levels(self, self._accepting_states)


class RewardAutomatonSimulator(DFASimulator, RewardSimulator):
    def __init__(self, dfa: RewardDFA):
        super().__init__(dfa)
        self.visited_states = {self.cur_state}

    def reset(self):
        super().reset()
        self.visited_states = {self.cur_state}

    def step(self, s: Set[Symbol], **kwargs):
        super().step(s)
        self.visited_states.add(self.cur_state)
        return self.cur_state

    def potential_function(self, q, is_terminal_state=False):
        return self.dfa.potential_function(q, is_terminal_state=is_terminal_state)

    def is_failed(self):
        return self.cur_state in self.dfa.failure_states


def _compute_levels(dfa: DFA, property_states):
    """
    Compute important details from
    :param dfa:
    :param property_states:
    :return:
    """
    assert property_states.issubset(dfa._states)
    level = 0
    state2level = {final_state: level for final_state in property_states}

    z_current = set()
    z_next = set(property_states)
    while z_current != z_next:
        level += 1
        z_current = z_next
        z_next = copy(z_current)
        for s in dfa._states:
            if s in z_current:
                continue
            for a in dfa._transition_function.get(s, []):
                next_state = dfa._transition_function[s][a]
                if next_state in z_current:
                    z_next.add(s)
                    state2level[s] = level

    z_current = z_next

    max_level = level - 1

    # levels for failure state (i.e. that cannot reach a final state)
    failure_states = set()
    for s in filter(lambda x: x not in z_current, dfa._states):
        state2level[s] = level
        failure_states.add(s)

    return state2level, max_level, failure_states


def _potential_function(q, initial_state, reachability_levels, reward, is_terminal_state=False):
    if is_terminal_state:
        return 0
    else:
        # p = 1/(self.reachability_levels[q]) * self.reward
        # if q == initial_state and reachability_levels[initial_state]==0:
        #     return reward
        initial_state_level = reachability_levels[initial_state]
        p = initial_state_level - reachability_levels[q]
        p = p/initial_state_level if initial_state_level!=0 else p
        p *= reward
    return p
