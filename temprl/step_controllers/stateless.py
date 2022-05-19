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

from typing import Callable

from temprl.step_controllers.base import AbstractStepController
from temprl.types import Interpretation


class StatelessStepController(AbstractStepController):
    """A class that allows to control the steps to be done by the temporal goals."""

    def __init__(
        self, step_func: Callable[[Interpretation], bool], allow_first: bool = True
    ):
        """
        Create the StepController.

        :param step_func: A function that takes a set of fluents and returns a boolean
        :param allow_first: If True, the first step always takes place
        """
        self.started = False
        self.step_func = step_func
        self.allow_first = allow_first

    def step(self, fluents: Interpretation) -> bool:
        """
        Update the step controller and check whether the step on the DFA can take place.

        :param: fluents: A set of fluents
        :return: True if the step can be taken, False otherwise
        """
        if self.allow_first and not self.started:
            # always allow the first step
            self.started = True
            return True
        if not self.started:
            # otherwise, if no step ever took place, check if it can start
            self.started = self.step_func(fluents)
            return self.started
        # else, simply check with the step function
        return self.step_func(fluents)

    def reset(self):
        """Reset the StepController."""
        self.started = False
