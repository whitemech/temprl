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

"""Tests for `temprl` package."""
from typing import Any, Dict, cast

import gym
import numpy as np
import sympy
from gym.spaces import Discrete, MultiDiscrete
from pythomata.impl.symbolic import SymbolicDFA

from temprl.reward_machines.automata import RewardAutomaton
from temprl.reward_machines.base import AbstractRewardMachine, RewardMachineSimulator
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper
from tests.utils import GymTestEnv, q_function_learn, q_function_test, wrap_observation


class TestSimpleEnv:
    """This class contains tests for a simple Gym environment."""

    env: gym.Env
    Q: Dict[Any, np.ndarray]

    @classmethod
    def setup_class(cls) -> None:
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)
        cls.Q = q_function_learn(cls.env, nb_episodes=200)

    def test_optimal_policy(self) -> None:
        """Test that the optimal policy maximizes the reward."""
        history = q_function_test(self.env, self.Q, nb_episodes=10)
        assert np.isclose(np.average(history), 1.0)

    @classmethod
    def teardown_class(cls) -> None:
        """Tear the tests down."""


class TestTempRLWithSimpleEnv:
    """This class contains tests for a Gym wrapper with a simple temporal goal."""

    automaton: SymbolicDFA
    reward: float
    reward_machine: AbstractRewardMachine
    env: gym.Env
    tg: TemporalGoal
    wrapped: TemporalGoalWrapper

    @classmethod
    def _build_automaton(cls) -> SymbolicDFA:
        """
        Build the reward automaton.

        It is equivalent to the following regular expression:

            (!s4)*;s3;(!s4)*;s0;(!s4)*;s4;true*

        :returns: the automaton.
        """
        automaton = SymbolicDFA()
        q0 = 0
        q1 = automaton.create_state()
        q2 = automaton.create_state()
        q3 = automaton.create_state()

        automaton.add_transition((q0, "~s4 & ~s3", q0))
        automaton.add_transition((q0, "s3", q1))
        automaton.add_transition((q1, "~s4 & ~s0", q1))
        automaton.add_transition((q1, "s0", q2))
        automaton.add_transition((q2, "~s4", q2))
        automaton.add_transition((q2, "s4", q3))
        automaton.add_transition((q3, "true", q3))
        automaton.set_accepting_state(q3, True)

        automaton = automaton.complete()

        return automaton

    @classmethod
    def setup_class(cls) -> None:
        """Set the tests up."""
        cls.automaton = cls._build_automaton()
        cls.reward = 10.0
        cls.reward_machine = RewardAutomaton(cls.automaton, cls.reward)
        cls.env = GymTestEnv(n_states=5)
        cls.tg = TemporalGoal(
            reward_machine=cls.reward_machine,
        )
        cls.wrapped = TemporalGoalWrapper(
            env=cls.env,
            temp_goals=[cls.tg],
            fluent_extractor=lambda obs, action: {"s" + str(obs)},
        )
        # from (s, [q]) to (s, q)
        n, q = cast(Discrete, cls.env.observation_space).n, cls.tg.observation_space.n
        cls.wrapped = wrap_observation(
            cls.wrapped, MultiDiscrete([n, q]), lambda o: (o[0], o[1][0])
        )

    def test_observation_space(self) -> None:
        """Test that the combined observation space is computed as expected."""
        assert MultiDiscrete([5, 5]) == self.wrapped.observation_space

    def test_temporal_goal_reward(self) -> None:
        """Test that the 'reward' property of the temporal goal works correctly."""
        assert cast(RewardAutomaton, self.reward_machine).reward == 10.0

    def test_temporal_goal_automaton(self) -> None:
        """Test that the 'automaton' property of the temporal goal works correctly."""
        assert isinstance(self.tg.automaton, AbstractRewardMachine)
        assert self.tg.automaton.get_transitions_from(0) == {
            (0, sympy.parse_expr("s4 & ~s3"), 4),
            (0, sympy.parse_expr("~s4 & ~s3"), 0),
            (0, sympy.parse_expr("s3"), 1),
        }
        self.tg.reset()
        assert self.tg.current_state == self.reward_machine.initial_state

    def test_learning_wrapped_env(self) -> None:
        """Test that learning with the unwrapped env is feasible."""
        Q = q_function_learn(self.wrapped, nb_episodes=5000, eps=0.5)
        history = q_function_test(self.wrapped, Q, nb_episodes=10)
        print(f"Reward history: {history}")
        assert np.isclose(np.average(history), 11.0)

    def test_reward_machine_get_transitions(self) -> None:
        """Test AbstractRewardMachine.get_transitions."""
        transitions = self.reward_machine.get_transitions()
        assert transitions == {
            (0, sympy.parse_expr("~s4 & ~s3"), 0),
            (0, sympy.parse_expr("s3"), 1),
            (1, sympy.parse_expr("~s4 & ~s0"), 1),
            (1, sympy.parse_expr("s0"), 2),
            (2, sympy.parse_expr("~s4"), 2),
            (2, sympy.parse_expr("s4"), 3),
            (3, sympy.parse_expr("true"), 3),
            (1, sympy.parse_expr("s4 & ~s0"), 4),
            (0, sympy.parse_expr("s4 & ~s3"), 4),
            (4, sympy.parse_expr("true"), 4),
        }

    def test_reward_machine_simulator_getters(self) -> None:
        """Test RewardMachineSimulator getters."""
        simulator = RewardMachineSimulator(self.reward_machine)
        assert simulator.reward_machine == self.reward_machine
        assert simulator.current_state == self.reward_machine.initial_state

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""
