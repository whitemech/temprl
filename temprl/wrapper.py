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
from typing import Callable, List, Optional, Tuple

import gym
from gym.spaces import Discrete, MultiDiscrete
from gym.spaces import Tuple as GymTuple

from temprl.reward_machines.base import AbstractRewardMachine, RewardMachineSimulator
from temprl.types import FluentExtractor, Interpretation, State

logger = logging.getLogger(__name__)


class TemporalGoal:
    """Abstract class to represent a temporal goal."""

    def __init__(
        self,
        reward_machine: AbstractRewardMachine,
    ):
        """
        Initialize a temporal goal.

        :param reward_machine: the reward
        """
        self._reward_machine = reward_machine
        self._simulator = RewardMachineSimulator(
            reward_machine,
        )

    @property
    def observation_space(self) -> Discrete:
        """Return the observation space of the temporal goal."""
        return Discrete(len(self._reward_machine.states))

    @property
    def automaton(self) -> AbstractRewardMachine:
        """Get the automaton."""
        return self._reward_machine

    def reset(self, initial_obs: Interpretation) -> State:
        """
        Reset the simulator.

        :param initial_obs: the fluents in the initial state.
        :return: the temporal goal state.
        """
        return self._simulator.reset(initial_obs)

    def step(self, symbol: Interpretation) -> Tuple[State, float]:
        """
        Do a step.

        :param symbol: the symbol to read.
        :return: the generated reward signal.
        """
        return self._simulator.step(symbol)

    def current_dfa_state(self) -> State:
        """Get the current DFA state."""
        return self._simulator.current_state


class StepController:
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

    def check(self, fluents: Interpretation) -> bool:
        """
        Check if the step on the DFA can take place.

        :param: fluents: A set of fluents
        :return: True if the step can be taken, False otherwise
        """
        if self.allow_first and not self.started:
            # always allow the first step
            self.started = True
            return True
        elif not self.started:
            # otherwise, if no step ever took place, check if it can start
            self.started = self.step_func(fluents)
            return self.started
        else:
            # else, simply check with the step function
            return self.step_func(fluents)

    def reset(self):
        """Reset the StepController."""
        self.started = False


class TemporalGoalWrapper(gym.Wrapper):
    """Gym wrapper to include a temporal goal in the environment."""

    def __init__(
        self,
        env: gym.Env,
        temp_goals: List[TemporalGoal],
        fluent_extractor: FluentExtractor,
        step_controller: Optional[StepController] = None,
    ):
        """
        Wrap a Gym environment with a temporal goal.

        :param env: the Gym environment to wrap.
        :param temp_goals: the temporal goal to be learnt
        :param fluent_extractor: the extractor of the fluents.
          A callable that takes in input an observation and the last action
          taken, and returns the set of fluents true in the current state.
        :param step_controller: the step controller that decides when a
          transition to the DFA has to take place.
        """
        super().__init__(env)
        self.temp_goals = temp_goals
        self.fluent_extractor: FluentExtractor = fluent_extractor
        self.step_controller = (
            step_controller
            if step_controller
            else StepController(step_func=lambda fluents: True, allow_first=True)
        )
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self) -> gym.spaces.Space:
        """Return the observation space."""
        temp_goals_shape = tuple(tg.observation_space.n for tg in self.temp_goals)
        return GymTuple(
            (self.env.observation_space, MultiDiscrete(list(temp_goals_shape)))
        )

    def step(self, action):
        """Do a step in the Gym environment."""
        obs, reward, done, info = super().step(action)
        fluents = self.fluent_extractor(obs, action)
        states_and_rewards = [
            tg.step(fluents)
            if self.step_controller.check(fluents)
            else (tg.current_dfa_state(), 0.0)
            for tg in self.temp_goals
        ]
        next_automata_states, temp_goal_rewards = zip(*states_and_rewards)
        total_goal_rewards = sum(temp_goal_rewards)
        obs_prime = (obs, next_automata_states)
        reward_prime = reward + total_goal_rewards
        return obs_prime, reward_prime, done, info

    def reset(self, **_kwargs):
        """Reset the Gym environment."""
        obs = super().reset()
        fluents = self.fluent_extractor(obs, None)
        automata_states = [tg.reset(fluents) for tg in self.temp_goals]
        return obs, automata_states
