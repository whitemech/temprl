from abc import ABC, abstractmethod

from pythogic.base.Alphabet import Alphabet
from pythogic.base.Formula import Formula

from pythomata.base.Simulator import Simulator
from pythomata.base.Symbol import Symbol

from rltg.agents.feature_extraction import FeatureExtractor
from rltg.logic.RewardAutomaton import RewardAutomaton
from rltg.logic.RewardAutomatonSimulator import RewardAutomatonSimulator


class TemporalEvaluator(ABC):
    def __init__(self, goal_feature_extractor:FeatureExtractor, alphabet:Alphabet, formula:Formula, reward):
        self.goal_feature_extractor = goal_feature_extractor
        self.alphabet = alphabet
        self.formula = formula
        self._automaton = RewardAutomaton._fromFormula(alphabet, formula, reward)
        self.simulator = RewardAutomatonSimulator(self._automaton)

    @abstractmethod
    def fromFeaturesToPropositional(self, features) -> Symbol:
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
