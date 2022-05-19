#
# Copyright 2020-2022 Marco Favorito
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
from typing import AbstractSet

from pythomata.core import DFA

from temprl.reward_machines.base import AbstractRewardMachine
from temprl.types import Interpretation, State, TransitionType


class RewardAutomaton(AbstractRewardMachine):
    """This class implements the reward automaton using a pythomata.DFA object."""

    def __init__(self, dfa: DFA, reward):
        """Initialize the reward automaton."""
        super().__init__()
        self._automaton = dfa
        self._reward = reward

    @property
    def states(self) -> AbstractSet[State]:
        """
        Get the set of states.

        :return: the set of states of the automaton.
        """
        return self._automaton.states

    @property
    def initial_state(self) -> State:
        """
        Get the initial state.

        :return: the initial state.
        """
        return self._automaton.initial_state

    def get_successor(self, state: State, symbol: Interpretation) -> State:
        """
        Get the (unique) successor.

        :param: state: the starting state.
        :param: symbol: the symbol to read.
        :returns: the next state. If not defined, return None.
        """
        dfa_symbol = {symbol_name: True for symbol_name in symbol}
        return self._automaton.get_successor(state, dfa_symbol)

    @property
    def accepting_states(self) -> AbstractSet[State]:
        """
        Get the set of accepting states.

        :return: the set of accepting states.
        """
        return self._automaton.accepting_states

    def get_transitions_from(self, state: State) -> AbstractSet[TransitionType]:
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

    def get_reward(self, state: State, symbol: Interpretation) -> float:
        """
        Get the reward associated to the transition.

        This method returns the reward only if the end state is accepting,
        because the automaton is an acceptor and the reward only depends
        on the acceptance condition of the destination state.

        :param state: the starting state.
        :param symbol: the read symbol.
        :return: the reward signal.
        """
        end_state = self.get_successor(state, symbol)
        return self.reward if self._automaton.is_accepting(end_state) else 0.0
