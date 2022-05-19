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

"""Base classes and interfaces for reward machines."""
from abc import ABC, ABCMeta, abstractmethod
from typing import AbstractSet, Tuple, cast

from temprl.helpers import enforce
from temprl.types import Interpretation, State, TransitionType


class _MetaRewardMachine(ABCMeta):
    """A metaclass for the AbstractRewardMachine class."""

    def __call__(cls, *args, **kwargs):
        """Init the subclass object."""
        enforce(
            issubclass(cls, AbstractRewardMachine),
            "this class cannot be the metaclass of a class that is not a subclass of 'Formula'",
            ValueError,
        )
        instance = cast(AbstractRewardMachine, super().__call__(*args, **kwargs))

        cls._check_initial_states_in_states(instance)
        return instance

    @classmethod
    def _check_initial_states_in_states(
        mcs, instance: "AbstractRewardMachine"
    ):  # pylint: disable=bad-mcs-classmethod-argument
        """Check the initial state of a RM is in the set of states."""
        enforce(
            instance.initial_state in instance.states,
            f"initial state {instance.initial_state} not in the set of states",
        )


class AbstractRewardMachine(ABC, metaclass=_MetaRewardMachine):
    """Interface for reward machines."""

    @property
    @abstractmethod
    def states(self) -> AbstractSet[State]:
        """Get the set of states."""

    @property
    @abstractmethod
    def initial_state(self) -> State:
        """Get the initial state."""

    @abstractmethod
    def get_transitions_from(self, state: State) -> AbstractSet[TransitionType]:
        """
        Get the outgoing transitions from a state.

        A transition is a triple (source_state, guard, destination_state).

        :param state: the source state.
        :return: the set of transitions object associated with that triple.
        :raise ValueError: if the state does not belong to the automaton.
        """

    @abstractmethod
    def get_reward(self, state: State, symbol: Interpretation) -> float:
        """
        Get the reward associated to the transition.

        :param state: the starting state.
        :param symbol: the read symbol.
        :return: the reward signal.
        :raise ValueError: if the provided state does not belong to the automaton.
        """

    @abstractmethod
    def get_successor(self, state: State, symbol: Interpretation) -> State:
        """
        Get the (unique) successor.

        :param state: the starting state.
        :param symbol: the read symbol.
        :return: the successor state.
        :raise ValueError: if the provided state does not belong to the automaton.
        """

    def get_transitions(self) -> AbstractSet[TransitionType]:
        """
        Get all the transitions.

        :return: the set of transitions.
        """
        transitions = set()
        for state in self.states:
            for start_state, guard, end_state in self.get_transitions_from(state):
                transitions.add((start_state, guard, end_state))
        return transitions


class AbstractRewardMachineSimulator(ABC):
    """Interface for abstract reward machine simulator."""

    @abstractmethod
    def reset(self, initial_obs: Interpretation) -> State:
        """
        Reset the simulation to its initial state.

        :param initial_obs: the fluents in the initial state.
        :return: the initial state.
        """

    @abstractmethod
    def step(self, symbol: Interpretation) -> Tuple[State, float]:
        """
        Do a step.

        :param symbol: the symbol to read.
        :return: the new state and the generated reward signal.
        """


class RewardMachineSimulator(AbstractRewardMachineSimulator):
    """Concrete class of AbstractRewardMachineSimulator."""

    def __init__(self, reward_machine: AbstractRewardMachine):
        """Initialize the reward machine simulator."""
        self._reward_machine = reward_machine
        self._current_state: State = self._reward_machine.initial_state

    @property
    def reward_machine(self) -> AbstractRewardMachine:
        """Get the simulated reward machine."""
        return self._reward_machine

    @property
    def current_state(self) -> State:
        """Get the current state."""
        return self._current_state

    def reset(self, initial_obs: Interpretation) -> State:
        """
        Reset the simulation to its initial state.

        :param initial_obs: the fluents in the initial state.
        :return: the initial state.
        """
        self._current_state = self._reward_machine.initial_state
        next_state, _reward = self.step(initial_obs)
        self._current_state = next_state
        return next_state

    def step(self, symbol: Interpretation) -> Tuple[State, float]:
        """
        Do a step.

        :param symbol: the symbol to read.
        :return: the new state and the generated reward signal.
        """
        reward = self._reward_machine.get_reward(self._current_state, symbol)
        next_state = self._reward_machine.get_successor(self._current_state, symbol)
        self._current_state = next_state
        return next_state, reward
