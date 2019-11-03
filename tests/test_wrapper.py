# -*- coding: utf-8 -*-
"""This module contains tests for the temprl/wrapper.py module."""
import pytest
from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation

from temprl.automata import RewardDFA
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper
from conftest import GymTestEnv


class TestWrapper:
    """Test that the wrapper works as expected."""

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestEnv(n_states=5)

        cls.formula = LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt")
        cls.tg = TemporalGoal(
            formula=cls.formula,
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            extract_fluents=None
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg], feature_extractor=None)

    def test_temporal_goal_reward(self):
        """Test that the 'reward' property of the temporal goal works correctly."""
        assert self.tg.reward == 10.0

    def test_temporal_goal_formula(self):
        """Test that the 'formula' property of the temporal goal works correctly."""
        assert self.tg.formula == self.formula

    def test_temporal_goal_automaton(self):
        """Test that the 'automaton' property of the temporal goal works correctly."""
        assert isinstance(self.tg.automaton, RewardDFA)

    def test_extract_fluents_raises_exception(self):
        """Test that the 'extract_fluents' property raises 'NotImplementedError' if not set in the constructor."""
        with pytest.raises(NotImplementedError):
            self.tg.extract_fluents(None, None)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""
