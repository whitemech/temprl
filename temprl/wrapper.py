# -*- coding: utf-8 -*-

"""Main module."""

from abc import abstractmethod, ABC
from typing import Optional, Set, List, Callable

import gym
import numpy as np
from flloat.semantics import PLInterpretation
from pythomata.base import Symbol, State

from temprl.automata import (
    TemporalLogicFormula,
    RewardDFA,
    RewardAutomatonSimulator
)


class TemporalGoal(ABC):
    """Abstract class to represent a temporal goal."""

    def __init__(
        self,
        formula: TemporalLogicFormula,
        reward: float,
        labels: Optional[Set[Symbol]] = None,
        reward_shaping: bool = True
    ):
        """
        Initialize a temporal goal.

        :param formula: the formula to be satisfied.
        :param reward: the reward associated to the temporal goal.
        :param labels: the set of all possible fluents
                     | (used to generate the full automaton).
        :param reward_shaping: the set of all possible fluents
                             | (used to generate the full automaton).
        """
        self._formula = formula
        self._automaton = RewardDFA.from_formula(
            self._formula,
            reward,
            alphabet=labels
        )
        self._simulator = RewardAutomatonSimulator(
            self._automaton,
            reward_shaping=reward_shaping
        )
        self._reward = reward

    @property
    def formula(self):
        """Get the formula."""
        return self._formula

    @property
    def automaton(self):
        """Get the automaton."""
        return self._automaton

    @property
    def reward(self):
        """Get the reward."""
        return self._reward

    @abstractmethod
    def extract_fluents(self, obs, action) -> PLInterpretation:
        """
        Extract high-level features from the observation.

        :return: the list of active fluents.
        """

    def step(self, observation, action) -> Optional[State]:
        """Do a step in the simulation."""
        fluents = self.extract_fluents(observation, action)
        self._simulator.step(fluents)
        return self._simulator.cur_state

    def reset(self):
        """Reset the simulation."""
        self._simulator.reset()
        return self._simulator.cur_state

    def observe_reward(self, is_terminal_state: bool = False) -> float:
        """Observe the reward of the last transition."""
        return self._simulator.observe_reward(is_terminal_state)

    def is_true(self):
        """Check if the simulation is in a final state."""
        return self._simulator.is_true()

    def is_failed(self):
        """Check whether the simulation has failed."""
        return self._simulator.is_failed()


class TemporalGoalWrapper(gym.Wrapper):
    """Gym wrapper to include a temporal goal in the environment."""

    def __init__(
        self,
        env: gym.Env,
        temp_goals: List[TemporalGoal],
        feature_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        """
        Wrap a Gym environment with a temporal goal.

        :param env: the Gym environment to wrap.
        :param temp_goals: the temporal goal to be learnt
        :param feature_extractor: (optional) extract feature
                                | from the environment state
        """
        super().__init__(env)
        self.temp_goals = temp_goals
        self.feature_extractor = feature_extractor\
            if feature_extractor is not None else (lambda x: x)

    def step(self, action):
        """Do a step in the Gym environment."""
        obs, reward, done, info = super().step(action)
        features = self.feature_extractor(obs)
        next_automata_states = [tg.step(obs, action) for tg in self.temp_goals]

        temp_goal_all_true = all(tg.is_true() for tg in self.temp_goals)
        temp_goal_some_false = any(tg.is_failed() for tg in self.temp_goals)
        done = done or temp_goal_all_true or temp_goal_some_false
        temp_goal_rewards = sum(
            tg.observe_reward(is_terminal_state=done)
            for tg in self.temp_goals
        )

        obs_prime = np.concatenate([features, next_automata_states])
        reward_prime = reward + temp_goal_rewards
        return obs_prime, reward_prime, done, info

    def reset(self, **kwargs):
        """Reset the Gym environment."""
        obs = super().reset()
        for tg in self.temp_goals:
            tg.reset()

        features = self.feature_extractor(obs)
        automata_states = [tg.reset() for tg in self.temp_goals]
        return np.concatenate([features, automata_states])
