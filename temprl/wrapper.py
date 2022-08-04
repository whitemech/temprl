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

"""Main module."""
import logging
from typing import List, Optional, Tuple

import gym
from gym.core import ActType
from gym.spaces import Discrete, MultiDiscrete
from gym.spaces import Tuple as GymTuple

from temprl.reward_machines.base import AbstractRewardMachine, RewardMachineSimulator
from temprl.step_controllers.base import AbstractStepController
from temprl.step_controllers.stateless import StatelessStepController
from temprl.types import FluentExtractor, Interpretation, Observation, State

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

    @property
    def current_state(self) -> State:
        """Get the current state."""
        return self._simulator.current_state

    def reset(self) -> None:
        """
        Reset the simulator.

        :return: the temporal goal state.
        """
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
        step_controller: Optional[AbstractStepController] = None,
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
            else StatelessStepController(
                step_func=lambda fluents: True, allow_first=True
            )
        )
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self) -> gym.spaces.Space:
        """Return the observation space."""
        temp_goals_shape = tuple(tg.observation_space.n for tg in self.temp_goals)
        return GymTuple(
            (self.env.observation_space, MultiDiscrete(list(temp_goals_shape)))
        )

    def step(self, action: ActType) -> Tuple[Observation, float, bool, dict]:
        """Do a step in the Gym environment."""
        obs, reward, done, info = super().step(action)
        fluents = self.fluent_extractor(obs, action)
        states_and_rewards = [
            tg.step(fluents)
            if self.step_controller.step(fluents)
            else (tg.current_state, 0.0)
            for tg in self.temp_goals
        ]
        next_automata_states, temp_goal_rewards = zip(*states_and_rewards)
        total_goal_rewards = sum(temp_goal_rewards)
        obs_prime = (obs, next_automata_states)
        reward_prime = reward + total_goal_rewards
        return obs_prime, reward_prime, done, info

    def reset(self, **kwargs) -> Observation:
        """
        Reset the Gym environment.

        The reward machine is synchronized starting from the second state of the trajectory.

        :param kwargs: the keyword arguments of the reset function.
        :return: the new initial state.
        """
        obs = super().reset(**kwargs)
        for tg in self.temp_goals:
            tg.reset()
        automata_states = [tg.current_state for tg in self.temp_goals]
        self.step_controller.reset()
        return obs, automata_states
