from abc import ABC, abstractmethod
from typing import Set

from flloat.base.Alphabet import Alphabet
from flloat.syntax.ldlf import LDLfFormula
from gym.spaces import Discrete
from pythomata.base.Symbol import Symbol

from rltg.agents.feature_extraction import FeatureExtractor
from rltg.logic.CompleteRewardAutomaton import CompleteRewardAutomaton
from rltg.logic.PartialRewardAutomaton import PartialRewardAutomaton
from rltg.logic.RewardAutomatonSimulator import RewardAutomatonSimulator


class TemporalEvaluator(ABC):
    def __init__(self, goal_feature_extractor:FeatureExtractor, alphabet:Set[Symbol], formula:LDLfFormula, reward,
                 gamma=0.99, on_the_fly=False):
        self.goal_feature_extractor = goal_feature_extractor
        self.alphabet = Alphabet(alphabet)
        self.formula = formula
        self.reward = reward
        self.on_the_fly = on_the_fly
        if not on_the_fly:
            self._automaton = CompleteRewardAutomaton._fromFormula(alphabet, formula, reward, gamma=gamma)
            self.simulator = RewardAutomatonSimulator(self._automaton)
        else:
            self.simulator = PartialRewardAutomaton(self.alphabet, self.formula, reward, gamma=gamma)

    @abstractmethod
    def fromFeaturesToPropositional(self, features, action, *args, **kwargs) -> Set[Symbol]:
        raise NotImplementedError

    def update(self, action, state):
        """update the automaton.
        :param action: the action to reach the state
        :param state:  the new state of the MDP
        :returns (new_automaton_state, reward)"""
        features = self.goal_feature_extractor(state)
        propositional = self.fromFeaturesToPropositional(features, action)
        old_state = self.simulator.get_current_state()
        new_state = self.simulator.make_transition(propositional)
        return self.simulator.get_current_state()


    def get_state(self):
        return self.simulator.get_current_state()

    def get_state_space(self):
        if not self.on_the_fly:
            return Discrete(len(self.simulator.state2id))
        else:
            # estimate the state space size.
            # the estimate MUST overestimate the true size.
            # return Discrete(100)
            return None

    def reset(self):
        self.simulator.reset()

    def is_failed(self):
        return self.simulator.is_failed()

    def is_true(self):
        return self.simulator.is_true()

    def is_terminal(self):
        return self.is_true() or self.is_failed()
