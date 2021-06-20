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

"""This module contains a naive implementation of q-learning."""
from collections import defaultdict
from typing import Any, Dict

import gym
import numpy as np


def q_function_learn(
        env: gym.Env, nb_episodes=100, alpha=0.1, eps=0.1, gamma=0.9
) -> Dict[Any, np.ndarray]:
    """
    Learn a Q-function from a Gym env using vanilla Q-Learning.

    :param env: the environment
    :param nb_episodes: the number of episodes
    :param alpha: the learning rate
    :param eps: the epsilon parameter in eps-greedy exploration
    :param gamma: the discount factor
    :returns: the Q function, a dictionary from states to array of Q values for every action.
    """
    nb_actions = env.action_space.n
    Q: Dict[Any, np.ndarray] = defaultdict(
        lambda: np.random.randn(
            nb_actions,
        )
                * 0.01
    )

    def choose_action(state):
        if np.random.random() < eps:
            return np.random.randint(0, nb_actions)
        else:
            return np.argmax(Q[state])

    for _ in range(nb_episodes):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state)
            state2, reward, done, info = env.step(action)
            Q[state][action] += alpha * (
                    reward + gamma * np.max(Q[state2]) - Q[state][action]
            )
            state = state2

    return Q


def q_function_test(
        env: gym.Env, Q: Dict[Any, np.ndarray], nb_episodes=10
) -> np.ndarray:
    """
    Test a Q-function against a Gym env.

    :param env: the environment
    :param Q: the action-value function
    :param nb_episodes: the number of episodes
    :returns: a list of rewards collected for every episode.
    """
    rewards = np.array([])
    for _ in range(nb_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)
            total_reward += reward
            state = state2

        rewards = np.append(rewards, total_reward)
    return rewards
