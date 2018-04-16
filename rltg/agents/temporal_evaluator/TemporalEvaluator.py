from abc import ABC, abstractmethod

from flloat.base.Alphabet import Alphabet
from flloat.syntax.ldlf import LDLfFormula

from pythomata.base.Symbol import Symbol
from typing import Set

from rltg.agents.feature_extraction import FeatureExtractor
from rltg.logic.RewardAutomaton import RewardAutomaton
from rltg.logic.RewardAutomatonSimulator import RewardAutomatonSimulator


class TemporalEvaluator(ABC):
    def __init__(self, goal_feature_extractor:FeatureExtractor, alphabet:Set[Symbol], formula:LDLfFormula, reward,
                 on_the_fly=False):
        self.goal_feature_extractor = goal_feature_extractor
        self.alphabet = Alphabet(alphabet)
        self.formula = formula
        if not on_the_fly:
            self._automaton = RewardAutomaton._fromFormula(alphabet, formula, reward)
            self.simulator = RewardAutomatonSimulator(self._automaton)
        else:
            self.simulator = self.formula.to_automaton(alphabet, on_the_fly=True)

    @abstractmethod
    def fromFeaturesToPropositional(self, features) -> Set[Symbol]:
        raise NotImplementedError

    def update(self, state):
        """update the automaton.
        :returns (new_automaton_state, reward)"""
        features = self.goal_feature_extractor(state)
        propositional = self.fromFeaturesToPropositional(features)
        reward = self.simulator.make_transition(propositional)
        return self.simulator.cur_state, reward

    def get_state(self):
        return self.simulator.cur_state

    def reset(self):
        self.simulator.reset()
