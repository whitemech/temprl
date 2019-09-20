# -*- coding: utf-8 -*-

"""Classes that implement automata that give the rewards to the RL agent."""

from abc import abstractmethod, ABC
from copy import copy
from typing import Union, Optional, Set

from flloat.ldlf import LDLfFormula
from flloat.ltlf import LTLfFormula
from flloat.semantics import PLInterpretation
from pythomata.base import Symbol, State
from pythomata.dfa import DFA
from pythomata.simulator import DFASimulator, Simulator

TemporalLogicFormula = Union[LTLfFormula, LDLfFormula]


class RewardAutomaton(ABC):
    """Abstract class for an automaton that gives rewards."""

    @abstractmethod
    def potential_function(self, q, is_terminal_state=False):
        """Return the potential at a given state."""


class RewardSimulator(Simulator):
    """A simulator for a reward automaton."""

    @abstractmethod
    def is_failed(self) -> bool:
        """
        Tell if the simulation is failed.

        :return: True if the simulation is in a failure state, False otherwise.
        """

    @abstractmethod
    def observe_reward(self, is_terminal_state: bool = False) -> float:
        """
        Observe the reward according to the last transition.

        :param is_terminal_state: whether we are at the end of the RL episode.
        :return: the reward
        """


class RewardDFA(DFA, RewardAutomaton):
    """This class implements the reward automaton."""

    def __init__(self, dfa: DFA, reward):
        """Initialize the reward automaton."""
        dfa = dfa.complete().renumbering()
        super().__init__(
            set(dfa._states),
            set(dfa._alphabet),
            dfa._initial_state,
            set(dfa._accepting_states),
            dfa._transition_function
        )

        self.reward = reward
        self.reachability_levels, self.max_level,\
            self.failure_states = self._compute_levels()

    @classmethod
    def from_formula(
            cls,
            f: TemporalLogicFormula,
            reward,
            alphabet: Set[Symbol] = None
    ):
        """Return the reward automaton associated with the formula."""
        dfa = f.to_automaton(alphabet)
        return RewardDFA(dfa, reward)

    def potential_function(self, q: Optional[State], is_terminal_state=False):
        """Return the potential function to the given automaton state."""
        if is_terminal_state:
            return 0
        else:
            # p = 1/(self.reachability_levels[q]) * self.reward
            # if q == initial_state and reachability_levels[initial_state]==0:
            #     return reward
            initial_state_level = self.reachability_levels[self._initial_state]
            p = initial_state_level - self.reachability_levels[q]
            p = p / initial_state_level if initial_state_level != 0 else p
            p *= self.reward
        return p

    def _compute_levels(self):
        """Compute the levels for the potential function."""
        return _compute_levels(self, self._accepting_states)


class RewardAutomatonSimulator(DFASimulator, RewardSimulator):
    """A DFA simulator for a reward automaton."""

    def __init__(self, dfa: RewardDFA, reward_shaping: bool):
        """Initialize the reward DFA simulator."""
        super().__init__(dfa)
        self.dfa = dfa
        self.visited_states = {self._cur_state}
        self.reward_shaping = reward_shaping
        self._previous_state = None  # type: Optional[State]

    def reset(self):
        """Reset the simulator."""
        super().reset()
        self.visited_states = {self._cur_state}
        self._previous_state = None

    def step(self, s: PLInterpretation, **kwargs):
        """Do a step for the simulation.."""
        self._previous_state = self._cur_state
        super().step(s)
        self.visited_states.add(self._cur_state)
        if self._previous_state != self._cur_state:
            print("transition idxs: ", self._previous_state, self._cur_state)

    def observe_reward(self, is_terminal_state: bool = False) -> float:
        """Observe the reward of the last transition."""
        previous_potential = self.dfa.potential_function(
            self._previous_state,
            is_terminal_state=False
        )
        current_potential = self.dfa.potential_function(
            self._cur_state,
            is_terminal_state=is_terminal_state
        )
        reward = self.dfa.reward\
            if is_terminal_state and self.is_true() else 0.0
        return reward + current_potential - previous_potential

    def is_failed(self):
        """Check if the simulation is failed."""
        return super().is_failed()\
            or self._cur_state in self.dfa.failure_states


def _compute_levels(dfa: DFA, property_states):
    """Compute details from the DFA.

    The details are:
    - distance to the goal
    - maximum distance
    - reachability

    :param dfa: the
    :param property_states:
    :return: Three values:
           | - a dictionary from the automaton states to
           |   the distance from any goal
           | - the maximum distance of a state from any goal
           | - the set of failure states.
    """
    assert property_states.issubset(dfa._states)
    level = 0
    state2level = {final_state: level for final_state in property_states}

    z_current = set()  # type: Set
    z_next = set(property_states)
    while z_current != z_next:
        level += 1
        z_current = z_next
        z_next = copy(z_current)
        for s in dfa._states:
            if s in z_current:
                continue
            for a in dfa.transition_function.get(s, []):
                next_state = dfa.transition_function[s][a]
                if next_state in z_current:
                    z_next.add(s)
                    state2level[s] = level

    z_current = z_next

    max_level = level - 1

    # levels for failure state (i.e. that cannot reach a final state)
    failure_states = set()
    for s in filter(lambda x: x not in z_current, dfa._states):
        state2level[s] = level
        failure_states.add(s)

    return state2level, max_level, failure_states
