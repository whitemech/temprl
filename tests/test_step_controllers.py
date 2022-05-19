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

"""Tests for `temprl.step_controllers` package."""
from pythomata.impl.symbolic import SymbolicDFA

from temprl.step_controllers.stateful import StatefulStepController
from temprl.step_controllers.stateless import StatelessStepController


def test_stateless_step_controller_when_not_started_and_not_allow_first() -> None:
    """Test StatelessStepController when allow_first=False and started=False."""
    sc = StatelessStepController(lambda fluents: fluents != set(), allow_first=False)
    # should not return True, since allow_first=False and fluents == set()
    assert not sc.step(set())


def test_stateful_step_controller() -> None:
    """Test StatefulStepController."""
    dfa = SymbolicDFA()
    dfa.create_state()

    dfa.add_transition((0, "a", 1))
    dfa.add_transition((0, "~a", 0))
    dfa.add_transition((1, "true", 1))
    dfa.set_accepting_state(1, True)

    sc = StatefulStepController(dfa)
    # if ~a, remains in the same state
    assert not sc.step(set())
    # if a, it goes to the accepting state
    assert sc.step({"a"})
    # any other step will return True
    assert sc.step(set())

    # after reset, we are in the initial state
    sc.reset()
    assert not sc.step(set())
