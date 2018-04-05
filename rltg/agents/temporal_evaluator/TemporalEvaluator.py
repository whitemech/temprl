from abc import ABC, abstractmethod

from pythogic.base.Alphabet import Alphabet
from pythogic.base.Formula import Formula

from pythomata.base.Simulator import Simulator
from pythomata.base.Symbol import Symbol

from rltg.agents.feature_extraction import FeatureExtractor
from rltg.logic.RewardAutomaton import RewardAutomaton


class TemporalEvaluator(ABC):
    def __init__(self, goal_feature_extractor:FeatureExtractor, alphabet:Alphabet, formula:Formula, reward):
        self.goal_feature_extractor = goal_feature_extractor
        self.alphabet = alphabet
        self.formula = formula
        self.automaton = RewardAutomaton(alphabet, formula, reward)
        self.simulator = Simulator(self.automaton)

    @abstractmethod
    def fromFeaturesToPropositional(self, features) -> Symbol:
        raise NotImplementedError

    def update(self, state):
        """update the automaton.
        :returns (automaton_state, reward)"""
        features = self.goal_feature_extractor(state)
        propositional = self.fromFeaturesToPropositional(features)
        self.simulator.make_transition(propositional)
        return self.simulator.cur_state, self.automaton.reward if self.simulator.is_true() else 0

    def reset(self):
        self.simulator.reset()
