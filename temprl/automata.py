# -*- coding: utf-8 -*-
#
# Copyright 2020 Marco Favorito
#
# ------------------------------
#
# This file is part of temprl.
#
# temprl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# temprl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with temprl.  If not, see <https://www.gnu.org/licenses/>.
#

"""Classes that implement automata that give the rewards to the RL agent."""
import logging
from abc import ABC, abstractmethod
from copy import copy
from typing import AbstractSet, Optional, Set, Union, cast

from flloat.ldlf import LDLfFormula
from flloat.ltlf import LTLfFormula
from pythomata.core import DFA, StateType, SymbolType
from pythomata.simulator import AbstractSimulator, AutomatonSimulator

logger = logging.getLogger(__name__)

TemporalLogicFormula = Union[LTLfFormula, LDLfFormula]


class RewardAutomaton(ABC):
    """Abstract class for an automaton that gives rewards."""

    @abstractmethod
    def potential_function(self, q, is_terminal_state=False):
        """Return the potential at a given state."""


class RewardSimulator(AbstractSimulator):
    """A simulator for a reward automaton."""

    @abstractmethod
    def is_failed(self) -> bool:
        """
        Tell if the simulation is failed.

        :return: True if the simulation is in a failure state, False otherwise.
        """

    @abstractmethod
    def observe_reward(self, is_terminal_state: bool = False) -> float:
        """
        Observe the reward according to the last transition.

        :param is_terminal_state: whether we are at the end of the RL episode.
        :return: the reward
        """


class RewardDFA(DFA, RewardAutomaton):
    """This class implements the reward automaton."""

    def __init__(self, dfa: DFA, reward):
        """Initialize the reward automaton."""
        self.dfa = dfa.complete()
        # TODO: renumbering removed
        # TODO: forward interface
        self._transition_function = self.dfa._transition_function
        self.to_graphviz = self.dfa.to_graphviz

        self.reward = reward
        (
            self.reachability_levels,
            self.max_level,
            self.failure_states,
        ) = self._compute_levels()

        self.sink_state = self._find_sink_state()

    def _find_sink_state(self) -> StateType:
        # TODO: assumption SymbolicDFA
        for s, delta in self._transition_function.items():
            if s in self.failure_states and all(s == t for t, _ in delta.items()):
                return s

    @property
    def states(self) -> AbstractSet[StateType]:
        """
        Get the set of states.
        :return: the set of states of the automaton.
        """
        return self.dfa.states

    @property
    def initial_state(self) -> StateType:
        """
        Get the initial state.
        :return: the initial state.
        """
        return self.dfa.initial_state

    @property
    def accepting_states(self) -> AbstractSet[StateType]:
        """
        Get the set of accepting states.
        :return: the set of accepting states.
        """
        return self.dfa.accepting_states

    def get_successors(
        self, state: StateType, symbol: SymbolType
    ) -> AbstractSet[StateType]:
        """
        Get the successors of the state in input when reading a symbol.
        :param state: the state from which to compute the successors.
        :param symbol: the symbol of the transition.
        :return: the set of successor states.
        :raise ValueError: if the state does not belong to the automaton, or the symbol is not correct.
        """
        return self.dfa.get_successors(state, symbol)

    def get_successor(
        self, state: StateType, symbol: SymbolType
    ) -> Optional[StateType]:
        """
        Get the (unique) successor.
        If not defined, return None.
        """
        return self.dfa.get_successor(state, symbol)

    @staticmethod
    def from_formula(f: TemporalLogicFormula, reward, alphabet: Set[SymbolType] = None):
        """Return the reward automaton associated with the formula."""
        dfa = f.to_automaton(alphabet)
        return RewardDFA(dfa, reward)

    def potential_function(self, q: Optional[StateType], is_terminal_state=False):
        """Return the potential function to the given automaton state."""
        if is_terminal_state or q is None:
            return 0
        else:
            initial_state_level = self.reachability_levels[self.initial_state]
            p = initial_state_level - self.reachability_levels[q]
            p = p / initial_state_level if initial_state_level != 0 else p
            p *= self.reward
        return p

    def _compute_levels(self):
        """Compute the levels for the potential function."""
        return _compute_levels(self, self.accepting_states)


class RewardAutomatonSimulator(AutomatonSimulator, RewardSimulator):
    """A DFA simulator for a reward automaton."""

    def __init__(
        self, dfa: RewardDFA, reward_shaping: bool, zero_terminal_state: bool = True
    ):
        """Initialize the reward DFA simulator."""
        super().__init__(dfa)
        self.dfa = dfa
        self.visited_states = {self._cur_state}
        self.reward_shaping = reward_shaping
        self.zero_terminal_state = zero_terminal_state
        self._previous_state = None

    def reset(self):
        """Reset the simulator."""
        super().reset()
        self.visited_states = {self._cur_state}
        self._previous_state = None

    def step(self, s: Set[str], **_kwargs):
        """Do a step for the simulation.."""
        self._previous_state = self._cur_state
        super().step(s)
        if self._cur_state is None:
            self._cur_state = self.dfa.sink_state
        self.visited_states.add(self._cur_state)
        if self._previous_state != self._cur_state:
            logger.debug(
                "transition idxs: {}, {}".format(self._previous_state, self._cur_state)
            )

    @property
    def _cur_state(self):
        return list(self._current_states)[0] if len(self._current_states) > 0 else None

    def observe_reward(self, is_terminal_state: bool = False) -> float:
        """Observe the reward of the last transition."""
        reward = self.dfa.reward if is_terminal_state and self.is_true() else 0.0

        if self.reward_shaping:
            previous_potential = self.dfa.potential_function(
                self._previous_state, is_terminal_state=False
            )
            current_potential = self.dfa.potential_function(
                self._cur_state,
                is_terminal_state=is_terminal_state and self.zero_terminal_state,
            )
            return (
                current_potential
                - previous_potential
                + (reward if is_terminal_state and self.zero_terminal_state else 0)
            )
        else:
            return reward

    def is_failed(self):
        """Check if the simulation is failed."""
        return super().is_failed() or self._cur_state in self.dfa.failure_states


def _compute_levels(dfa: DFA, property_states):
    """Compute details from the DFA.

    The details are:
    - distance to the goal
    - maximum distance
    - reachability

    :param dfa: the
    :param property_states:
    :return: Three values:
           | - a dictionary from the automaton states to
           |   the distance from any goal
           | - the maximum distance of a state from any goal
           | - the set of failure states.
    """
    assert property_states.issubset(dfa.states)
    level = 0
    state2level = {final_state: level for final_state in property_states}

    z_current = set()  # type: Set
    z_next = set(property_states)
    while z_current != z_next:
        level += 1
        z_current = z_next
        z_next = copy(z_current)
        for s in dfa.states:
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
    for s in filter(lambda x: x not in z_current, dfa.states):
        state2level[s] = level
        failure_states.add(s)

    return state2level, max_level, failure_states
