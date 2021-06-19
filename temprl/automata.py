# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Marco Favorito
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
from typing import AbstractSet, Optional, Tuple

from pythomata.core import DFA, StateType, SymbolType, TransitionType

from temprl.types import Interpretation, State

logger = logging.getLogger(__name__)


class AbstractRewardMachine(DFA):
    """
    Interface for reward machines.

    That is, the machine yields a reward whenever a transitions
    ends to an accepting state.
    """

    @abstractmethod
    def get_reward(
        self, start_state: State, symbol: Interpretation, end_state: State
    ) -> float:
        """
        Get the reward associated to the transition.

        :param start_state: the starting state.
        :param symbol: the read symbol.
        :param end_state: the state we end up in.
        :return: the reward signal.
        """


class AbstractRewardMachineSimulator(ABC):
    """Interface for abstract reward machine simulator."""

    @abstractmethod
    def reset(self) -> State:
        """
        Reset the simulation to its initial state.

        :return: the initial state.
        """

    @abstractmethod
    def step(self, symbol: Interpretation) -> Tuple[State, float]:
        """
        Do a step.

        :param symbol: the symbol to read.
        :return: the new state and the generated reward signal.
        """


class RewardDFA(AbstractRewardMachine):
    """This class implements the reward automaton using a pythomata.DFA object."""

    def __init__(self, dfa: DFA, reward):
        """Initialize the reward automaton."""
        super().__init__()
        self._automaton = dfa
        self._reward = reward
        self._sink_state = None
        assert self._sink_state not in self._automaton.states

    def get_successor(
        self, state: StateType, symbol: SymbolType
    ) -> Optional[StateType]:
        """
        Get the (unique) successor.

        :param: state: the starting state.
        :param: symbol: the symbol to read.
        :returns: the next state. If not defined, return None.
        """
        # SymbolicDFA wants a dictionary, not a set
        symbol = {symbol_name: True for symbol_name in symbol}
        return self._automaton.get_successor(state, symbol)

    @property
    def states(self) -> AbstractSet[StateType]:
        """
        Get the set of states.

        :return: the set of states of the automaton.
        """
        # we add one virtual state for the 'super' sink state
        # that is, when the symbol is not in the alphabet.
        return set(self._automaton.states).union({self._sink_state})

    @property
    def initial_state(self) -> StateType:
        """
        Get the initial state.

        :return: the initial state.
        """
        return self._automaton.initial_state

    @property
    def accepting_states(self) -> AbstractSet[StateType]:
        """
        Get the set of accepting states.

        :return: the set of accepting states.
        """
        return self._automaton.accepting_states

    def get_transitions_from(self, state: StateType) -> AbstractSet[TransitionType]:
        """
        Get the outgoing transitions from a state.

        A transition is a triple (source_state, guard, destination_state).

        :param: state: the source state.
        :returns: the set of transitions object associated with that triple.
        :raise: ValueError: if the state does not belong to the automaton.
        """
        return self._automaton.get_transitions_from(state)

    @property
    def reward(self) -> float:
        """Return the reward."""
        return self._reward

    def get_reward(
        self, _start_state: State, _symbol: Interpretation, end_state: State
    ) -> float:
        """
        Get the reward associated to the transition.

        This method returns the reward only if the end state is accepting,
        because the automaton is an acceptor and the reward only depends
        on the acceptance condition of the destination state.

        :param _start_state: the starting state. Not used in this implementation.
        :param _symbol: the read symbol.
        :param end_state: the state we end up in.
        :return: the reward signal.
        """
        return (
            0.0
            if not self._automaton.is_accepting(end_state)
            or end_state == self._sink_state
            else self.reward
        )


class RewardDFASimulator(AbstractRewardMachineSimulator):
    """A DFA simulator for a reward automaton."""

    def __init__(self, dfa: AbstractRewardMachine):
        """Initialize the reward DFA simulator."""
        self.reward_machine = dfa
        self._current_state: Optional[State] = self.reward_machine.initial_state

    def reset(self) -> State:
        """
        Reset the simulation to its initial state.

        :return: the initial state.
        """
        self._current_state = self.reward_machine.initial_state
        return self._current_state

    def step(self, symbol: Interpretation) -> Tuple[State, float]:
        """
        Do a step.

        :param symbol: the symbol to read.
        :return: the new state and the generated reward signal.
        """
        next_state = self.reward_machine.get_successor(self._current_state, symbol)
        reward = self.reward_machine.get_reward(self._current_state, symbol, next_state)
        self._current_state = next_state
        return next_state, reward
