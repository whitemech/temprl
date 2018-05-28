from abc import ABC, abstractmethod

import numpy as np
from typing import List

from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.utils.StatsManager import StatsManager


class StoppingCondition(ABC):
    @abstractmethod
    def check_condition(self, *args, stats_manager:StatsManager=None, **kwargs) -> bool:
        raise NotImplementedError

class StatsStoppingCondition(StoppingCondition):
    def __init__(self, window_size=100):
        assert window_size > 0
        self.window_size = window_size

    def check_condition(self, *args, stats_manager: StatsManager = None, **kwargs):
        if stats_manager is None:
            raise Exception
        if len(stats_manager.total_reward_history)<self.window_size:
            return False
        return True


class GoalPercentage(StatsStoppingCondition):
    def __init__(self, window_size=100, min_perc=0.97):
        super().__init__(window_size)
        assert 0.0 < min_perc <= 1.0
        self.min_perc = min_perc

    def check_condition(self, *args, stats_manager:StatsManager=None, **kwargs):
        super_check = super().check_condition(*args, stats_manager=stats_manager, **kwargs)
        goal_percentage = np.mean(stats_manager.goals[-self.window_size:])
        return super_check and goal_percentage >= self.min_perc

class AvgRewardPercentage(StatsStoppingCondition):
    def __init__(self, window_size=100, target_mean=1.0):
        super().__init__(window_size)
        self.target_mean = target_mean

    def check_condition(self, *args, stats_manager:StatsManager=None, **kwargs):
        super_check = super().check_condition(*args, stats_manager=stats_manager, **kwargs)

        avg_reward = np.mean(stats_manager.total_reward_history[-self.window_size:])
        return super_check and avg_reward >= self.target_mean


class CheckAutomataInFinalState(StoppingCondition):

    def check_condition(self, *args, temporal_evaluators:List[TemporalEvaluator]=[], **kwargs):
        if len(temporal_evaluators)==0:
            return True
        return all(t.simulator.is_true() for t in temporal_evaluators)

