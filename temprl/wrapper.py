# -*- coding: utf-8 -*-

"""Main module."""
import logging
from abc import ABC
from typing import Optional, Set, List, Callable, Any, Tuple

import gym
from flloat.semantics import PLInterpretation
from gym.spaces import Discrete, MultiDiscrete
from pythomata.base import Symbol, State
from pythomata.dfa import DFA

from temprl.automata import (
    TemporalLogicFormula,
    RewardDFA,
    RewardAutomatonSimulator
)

Observation = Any
Action = Any

logger = logging.getLogger(__name__)


class TemporalGoal(ABC):
    """Abstract class to represent a temporal goal."""

    def __init__(
        self,
        formula: Optional[TemporalLogicFormula] = None,
        reward: float = 1.0,
        automaton: Optional[DFA] = None,
        labels: Optional[Set[Symbol]] = None,
        reward_shaping: bool = True,
        extract_fluents: Optional[Callable] = None,
        zero_terminal_state: bool = False,
        one_hot_encoding: bool = False,
    ):
        """
        Initialize a temporal goal.

        :param formula: the formula to be satisfied. it will be ignored if automaton is set.
        :param automaton: the pythomata.DFA instance. it will be
                        | the preferred input against 'formula'.
        :param reward: the reward associated to the temporal goal.
        :param labels: the set of all possible fluents
                     | (used to generate the full automaton).
        :param reward_shaping: the set of all possible fluents
                             | (used to generate the full automaton).
        :param extract_fluents: a callable that takes an observation
                             | and an actions, and returns a
                             | propositional interpretation with the active fluents.
                             | if None, the 'extract_fluents' method is taken.
        :param zero_terminal_state: when reward_shaping is True, make the
                                  | potential function at a terminal state equal to zero.
        :param one_hot_encoding: use one-hot encoding for representing the
                               | automata dimensions.
        """
        if formula is None and automaton is None:
            raise ValueError("Provide either a formula or an automaton.")

        self._formula = formula
        if automaton:
            self._automaton = RewardDFA(automaton, reward)
        else:
            self._automaton = RewardDFA.from_formula(
                self._formula,
                reward,
                alphabet=labels
            )
        self._simulator = RewardAutomatonSimulator(
            self._automaton,
            reward_shaping=reward_shaping,
            zero_terminal_state=zero_terminal_state
        )
        self._reward = reward
        self._one_hot_encoding = one_hot_encoding
        if extract_fluents is not None:
            setattr(self, "extract_fluents", extract_fluents)

    @property
    def observation_space(self) -> Discrete:
        """Return the observation space of the temporal goal."""
        # we add one virtual state for the 'super' sink state
        # - that is, when the symbol is not in the alphabet.
        # This is going to be a temporary workaround due to
        # the Pythomata's lack of support for this corner case.
        return Discrete(len(self._automaton.states) + 1)

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

    def extract_fluents(self, obs, action) -> PLInterpretation:
        """
        Extract high-level features from the observation.

        :return: the list of active fluents.
        """
        raise NotImplementedError

    def step(self, observation, action) -> Optional[State]:
        """Do a step in the simulation."""
        fluents = self.extract_fluents(observation, action)
        self._simulator.step(fluents)

        result = self._simulator.cur_state if self._simulator.cur_state is not None\
            else len(self._simulator.dfa.states)
        return result

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
        feature_extractor: Optional[Callable[[Observation, Action], Any]] = None,
        combine: Optional[Callable[[Observation, Tuple], Any]] = None,
    ):
        """
        Wrap a Gym environment with a temporal goal.

        :param env: the Gym environment to wrap.
        :param temp_goals: the temporal goal to be learnt
        :param feature_extractor: (optional) extract feature
                                | from the environment state
        :param combine: (optional) combine the agent state with
                      | the temporal goal state.
        """
        super().__init__(env)
        self.temp_goals = temp_goals
        self.feature_extractor = feature_extractor\
            if feature_extractor is not None else (lambda obs, action: obs)
        self.combine = combine\
            if combine is not None else (lambda obs, qs: tuple((obs, *qs)))

        self.observation_space = self._get_observation_space()

    def _get_observation_space(self) -> gym.spaces.Space:
        """Return the observation space."""
        if isinstance(self.env.observation_space, MultiDiscrete):
            env_shape = tuple(self.env.observation_space.nvec)
        else:
            env_shape = (self.env.observation_space.n, )
        temp_goals_shape = tuple(tg.observation_space.n for tg in self.temp_goals)

        combined_obs_space = env_shape + temp_goals_shape
        return MultiDiscrete(combined_obs_space)

    def step(self, action):
        """Do a step in the Gym environment."""
        obs, reward, done, info = super().step(action)
        features = self.feature_extractor(obs=obs, action=action)
        next_automata_states = tuple([tg.step(obs, action) for tg in self.temp_goals])

        # temp_goal_all_true = all(tg.is_true() for tg in self.temp_goals)
        # temp_goal_some_false = any(tg.is_failed() for tg in self.temp_goals)
        temp_goal_rewards = [
            tg.observe_reward(is_terminal_state=done)
            for tg in self.temp_goals
        ]
        total_goal_rewards = sum(temp_goal_rewards)

        if any(r != 0.0 for r in temp_goal_rewards):
            logger.debug("Non-zero goal rewards: {}".format(temp_goal_rewards))

        obs_prime = self.combine(features, next_automata_states)
        reward_prime = reward + total_goal_rewards
        return obs_prime, reward_prime, done, info

    def reset(self, **kwargs):
        """Reset the Gym environment."""
        obs = super().reset()
        for tg in self.temp_goals:
            tg.reset()

        features = self.feature_extractor(obs, None)
        automata_states = tuple([tg.reset() for tg in self.temp_goals])
        new_observation = self.combine(features, automata_states)
        return new_observation
