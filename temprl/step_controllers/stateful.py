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

"""This module contains an implementation of a stateless step controller."""
from pythomata.core import DFA
from pythomata.simulator import AutomatonSimulator

from temprl.step_controllers.base import AbstractStepController
from temprl.types import Guard, Interpretation, State


class StatefulStepController(AbstractStepController):
    """A class that allows to control the steps to be done by the temporal goals."""

    def __init__(self, acceptor: DFA[State, Interpretation, Guard]):
        """
        Create the StepController.

        :param acceptor: a pythomata.DFA object.
        """
        self._acceptor = acceptor
        self._simulator = AutomatonSimulator(self._acceptor)

    def step(self, fluents: Interpretation) -> bool:
        """
        Update the step controller and check whether the step on the DFA can take place.

        :param: fluents: A set of fluents
        :return: True if the step can be taken, False otherwise
        """
        self._simulator.step({f: True for f in fluents})
        return self._simulator.is_true()

    def reset(self) -> None:
        """Reset the StepController."""
        self._simulator.reset()
