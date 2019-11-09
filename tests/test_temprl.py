# -*- coding: utf-8 -*-
"""Tests for `temprl` package."""

import numpy as np
from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal
from .conftest import GymTestEnv, q_function_learn, q_function_test


class TestSimpleEnv:
    """This class contains tests for a simple Gym environment."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)
        cls.Q = q_function_learn(cls.env, nb_episodes=100)

    def test_optimal_policy(self):
        """Test that the optimal policy maximizes the reward."""
        history = q_function_test(self.env, self.Q, nb_episodes=10)
        assert np.isclose(np.average(history), 1.0)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""


class TestTempRLWithSimpleEnv:
    """This class contains tests for a Gym wrapper with a simple temporal goal."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)
        cls.tg = TemporalGoal(
            formula=LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt"),
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            extract_fluents=lambda obs, action: PLInterpretation({"s" + str(obs)})
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg], feature_extractor=None)
        cls.Q = q_function_learn(cls.wrapped, nb_episodes=100)

    def test_learning_wrapped_env(self):
        """Test that learning with the unwrapped env is feasible."""
        history = q_function_test(self.wrapped, self.Q, nb_episodes=10)
        print("Reward history: {}".format(history))
        assert np.isclose(np.average(history), 11.0)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""
