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

"""Main module."""
import logging
from abc import ABC
from typing import Any, Callable, List, Optional, Tuple

import gym
from gym.spaces import Discrete, MultiDiscrete
from pythomata.core import DFA

from temprl.automata import RewardDFA, RewardDFASimulator
from temprl.types import FluentExtractor, Interpretation, State

Observation = Any
Action = Any

logger = logging.getLogger(__name__)


class TemporalGoal(ABC):
    """Abstract class to represent a temporal goal."""

    def __init__(
        self,
        reward: float,
        automaton: DFA = None,
    ):
        """
        Initialize a temporal goal.

        :param automaton: the pythomata.DFA instance. it will be
                        | the preferred input against 'formula'.
        :param reward: the reward associated to the temporal goal.
        """
        self._automaton = RewardDFA(automaton, reward)
        self._simulator = RewardDFASimulator(
            self._automaton,
        )
        self._reward = reward

    @property
    def observation_space(self) -> Discrete:
        """Return the observation space of the temporal goal."""
        # we add one virtual state for the 'super' sink state
        # - that is, when the symbol is not in the alphabet.
        # This is going to be a temporary workaround due to
        # the Pythomata's lack of support for this corner case.
        return Discrete(len(self._automaton.states) + 1)

    @property
    def automaton(self):
        """Get the automaton."""
        return self._automaton

    @property
    def reward(self):
        """Get the reward."""
        return self._reward

    def reset(self) -> State:
        """Reset the simulator."""
        return self._simulator.reset()

    def step(self, symbol: Interpretation) -> Tuple[State, float]:
        """
        Do a step.

        :param symbol: the symbol to read.
        :return: the generated reward signal.
        """
        return self._simulator.step(symbol)


class TemporalGoalWrapper(gym.Wrapper):
    """Gym wrapper to include a temporal goal in the environment."""

    def __init__(
        self,
        env: gym.Env,
        temp_goals: List[TemporalGoal],
        fluent_extractor: FluentExtractor,
        feature_extractor: Optional[Callable[[Observation, Action], Any]] = None,
        combine: Optional[Callable[[Observation, Tuple], Any]] = None,
    ):
        """
        Wrap a Gym environment with a temporal goal.

        :param env: the Gym environment to wrap.
        :param temp_goals: the temporal goal to be learnt
        :param fluent_extractor: the extractor of the fluents.
          A callable that takes in input an observation and the last action
          taken, and returns the set of fluents true in the current state.
        :param feature_extractor: (optional) extract feature
                                | from the environment state
        :param combine: (optional) combine the agent state with
                      | the temporal goal state.
        """
        super().__init__(env)
        self.temp_goals = temp_goals
        self.fluent_extractor: FluentExtractor = fluent_extractor
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else (lambda obs, action: obs)
        )
        self.combine = (
            combine if combine is not None else (lambda obs, qs: tuple((obs, *qs)))
        )
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self) -> gym.spaces.Space:
        """Return the observation space."""
        if isinstance(self.env.observation_space, MultiDiscrete):
            env_shape = tuple(self.env.observation_space.nvec)
        else:
            env_shape = (self.env.observation_space.n,)
        temp_goals_shape = tuple(tg.observation_space.n for tg in self.temp_goals)

        combined_obs_space = env_shape + temp_goals_shape
        return MultiDiscrete(combined_obs_space)

    def step(self, action):
        """Do a step in the Gym environment."""
        obs, reward, done, info = super().step(action)
        features = self.feature_extractor(obs=obs, action=action)
        fluents = self.fluent_extractor(obs, action)
        states_and_rewards = [tg.step(fluents) for tg in self.temp_goals]
        next_automata_states, temp_goal_rewards = zip(*states_and_rewards)
        total_goal_rewards = sum(temp_goal_rewards)
        obs_prime = self.combine(features, next_automata_states)
        reward_prime = reward + total_goal_rewards
        return obs_prime, reward_prime, done, info

    def reset(self, **_kwargs):
        """Reset the Gym environment."""
        obs = super().reset()
        for tg in self.temp_goals:
            tg.reset()

        features = self.feature_extractor(obs, None)
        automata_states = tuple([tg.reset() for tg in self.temp_goals])
        new_observation = self.combine(features, automata_states)
        return new_observation
