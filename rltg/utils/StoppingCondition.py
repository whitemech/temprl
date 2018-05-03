from abc import ABC, abstractmethod

import numpy as np
from typing import List

from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.utils.StatsManager import StatsManager


class StoppingCondition(ABC):
    @abstractmethod
    def check_condition(self, *args, stats_manager:StatsManager=None, **kwargs) -> bool:
        raise NotImplementedError

class GoalPercentage(StoppingCondition):
    def __init__(self, window_size=100, min_perc=0.97):
        assert window_size > 0
        assert 0.0 < min_perc <= 1.0
        self.window_size = window_size
        self.min_perc = min_perc

    def check_condition(self, *args, stats_manager:StatsManager=None, **kwargs):
        if stats_manager is None:
            raise Exception

        goal_percentage = np.mean(stats_manager.goals[-self.window_size:])
        return goal_percentage >= self.min_perc

class CheckAutomataInFinalState(StoppingCondition):

    def check_condition(self, *args, temporal_evaluators:List[TemporalEvaluator]=[], **kwargs):
        if len(temporal_evaluators)==0:
            return True
        return all(t.simulator.is_true() for t in temporal_evaluators)
