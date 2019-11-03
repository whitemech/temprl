# -*- coding: utf-8 -*-
"""Tests for `temprl` package."""
import math

import numpy as np
from conftest import GymTestObsWrapper
from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from gym.spaces import MultiDiscrete
from keras import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from temprl.wrapper import TemporalGoalWrapper, TemporalGoal


class TestSimpleEnv:
    """This class contains tests for a simple Gym environment."""

    @classmethod
    def _build_model(cls, env):
        nb_actions = env.action_space.n
        model = Sequential()
        model.add(Flatten(input_shape=(1, ) + env.observation_space.shape))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        return model

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestObsWrapper(n_states=5)

        cls.model = cls._build_model(cls.env)
        memory = SequentialMemory(limit=2000, window_length=1)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                      value_min=.05, value_test=.0, nb_steps=10000)
        cls.dqn = DQNAgent(model=cls.model, nb_actions=cls.env.action_space.n, memory=memory,
                           nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
        cls.dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        cls.dqn.fit(cls.env, nb_steps=15000, visualize=False, verbose=2)

    def test_best_action(self):
        """Test that a simple model learns the optimal actions."""
        assert self.dqn.forward((0, )) == 2
        assert self.dqn.forward((1, )) == 2
        assert self.dqn.forward((2, )) == 2
        assert self.dqn.forward((3, )) == 2

    def test_optimal_policy(self):
        """Test that the optimal policy maximizes the reward."""
        history = self.dqn.test(self.env, nb_episodes=10)
        assert np.isclose(np.average(history.history["episode_reward"][1:]), 1.0)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""


class TestTempRLWithSimpleEnv:
    """This class contains tests for a Gym wrapper with a simple temporal goal."""

    @classmethod
    def _build_model(cls, env):
        nb_actions = env.action_space.n
        model = Sequential()
        model.add(Flatten(input_shape=(10, ) + env.observation_space.shape))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        return model

    @classmethod
    def setup_class(cls):
        """Set the tests up."""
        cls.env = GymTestObsWrapper(n_states=5)
        cls.tg = TemporalGoal(
            formula=LDLfParser()("<(!s4)*;s3;(!s4)*;s0;(!s4)*;s4>tt"),
            reward=10.0,
            labels={"s0", "s1", "s2", "s3", "s4"},
            reward_shaping=True,
            extract_fluents=lambda obs, action: PLInterpretation({"s" + str(obs[0])})
        )
        cls.wrapped = TemporalGoalWrapper(env=cls.env, temp_goals=[cls.tg], feature_extractor=None)

    def test_observation_space(self):
        """Test that the combined observation space is computed as expected."""
        assert self.wrapped.observation_space == MultiDiscrete((5, 6))

    def test_reward_shaping(self):
        """Test that the reward shaping works as expected."""
        obs = self.wrapped.reset()
        assert np.array_equal(obs, [0, 0])
        assert not self.tg.is_true()
        assert not self.tg.is_failed()

        # s1
        obs, reward, done, info = self.wrapped.step(2)
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(2)
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(2)
        assert math.isclose(reward, 3.3333, rel_tol=1e-9, abs_tol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(0)
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(0)
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s0 - positive reward
        obs, reward, done, info = self.wrapped.step(0)
        assert math.isclose(reward, 3.3333, rel_tol=1e-9, abs_tol=0.0001)
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s1
        obs, reward, done, info = self.wrapped.step(2)
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s2
        obs, reward, done, info = self.wrapped.step(2)
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s3
        obs, reward, done, info = self.wrapped.step(2)
        assert reward == 0
        assert not self.tg.is_true()
        assert not self.tg.is_failed()
        # s4
        obs, reward, done, info = self.wrapped.step(2)
        assert math.isclose(reward, 4.3333, rel_tol=1e-9, abs_tol=0.0001)
        assert done
        assert not self.tg.is_failed()
        assert self.tg.is_true()

    def test_when_temporal_goal_is_failed(self):
        """Test the case when the temporal goal is failed."""
        obs = self.wrapped.reset()
        assert np.array_equal(obs, [0, 0])

        # s1
        obs, reward, done, info = self.wrapped.step(2)
        assert reward == 0
        # s2
        obs, reward, done, info = self.wrapped.step(2)
        assert reward == 0
        # s3 - positive reward
        obs, reward, done, info = self.wrapped.step(2)
        assert math.isclose(reward, 3.3333, rel_tol=1e-9, abs_tol=0.0001)
        # s4 - temporal goal fails.
        obs, reward, done, info = self.wrapped.step(2)
        assert math.isclose(reward, 1 - 3.3333, rel_tol=1e-9, abs_tol=0.0001)
        assert self.tg.is_failed()

    def test_learning_wrapped_env(self):
        """Test that learning with the unwrapped env is feasible."""
        self.model = self._build_model(self.wrapped)
        memory = SequentialMemory(limit=2000, window_length=10)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                      value_min=.05, value_test=0.0, nb_steps=10000)
        dqn = DQNAgent(model=self.model, nb_actions=3, memory=memory, nb_steps_warmup=5000,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        dqn.fit(self.wrapped, nb_steps=15000, visualize=False, verbose=2)

        history = dqn.test(self.wrapped, nb_episodes=10)
        print("Reward history: {}".format(history.history["episode_reward"]))
        assert np.isclose(np.average(history.history["episode_reward"][1:]), 11.0)

    @classmethod
    def teardown_class(cls):
        """Tear the tests down."""
