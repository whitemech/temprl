"""Summary:

#################################################################################################
FEATURE EXTRACTORS
#################################################################################################


Two FeatureExtractor:
 - BreakoutRobotFeatureExtractor:           Used by the robot for the main task.
                                            returns (ball_x, ball_y, ball_dir, paddle_x)

 - BreakoutGoalFeatureExtractor:            Used by the temporal goal evaluators.
                                            returns a matrix of booleans: columns * rows
                                            representing the state of each brick

#################################################################################################
TEMPORAL EVALUATORS
#################################################################################################

- BreakoutCompleteLinesTemporalEvaluator:   Abstract class for temporal goals which involve the ordered deletion of
                                            lines of bricks, rows or columns.
                                            Uses BreakoutGoalFeatureExtractor

- BreakoutCompleteRowsTemporalEvaluator:    extends from BreakoutCompleteLinesTemporalEvaluator
                                            deletion of rows bottom-up or top-down

- BreakoutCompleteColumnsTemporalEvaluator: extends from BreakoutCompleteLinesTemporalEvaluator
                                            deletion of columns from left-to-right or right-to-left

#################################################################################################
"""
from abc import abstractmethod

import numpy as np
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from gym.spaces import Box, Tuple
from rltg.agents.feature_extraction import FeatureExtractor, RobotFeatureExtractor
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator


class BreakoutRobotFeatureExtractor(RobotFeatureExtractor):
    """Feature Extractor for the robot.
    It contains: ball x, ball y, ball direction and paddle x"""

    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (paddle_x, ball_x, ball_y)
        robot_feature_space = Tuple((
            obs_space.spaces["ball_x"],
            obs_space.spaces["ball_y"],
            obs_space.spaces["ball_dir"],
            obs_space.spaces["paddle_x"],
        ))
        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["ball_x"],
                input["ball_y"],
                input["ball_dir"],
                input["paddle_x"])


class BreakoutGoalFeatureExtractor(FeatureExtractor):
    def __init__(self, obs_space, bricks_rows=3, bricks_cols=3):
        output_space = Box(low=0, high=1, shape=(bricks_cols, bricks_rows), dtype=np.uint8)
        super().__init__(obs_space, output_space)

    def _extract(self, input, **kwargs):
        return input["bricks_matrix"]


class BreakoutCompleteLinesTemporalEvaluator(TemporalEvaluator):
    """Breakout temporal evaluator for delete columns from left to right"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3):
        self.line_symbols = [Symbol("l%s" % i) for i in range(bricks_rows)]
        lines = self.line_symbols

        parser = LDLfParser()
        f = parser("<(!l0 & !l1 & !l2)*;(l0 & !l1 & !l2)*;(l0 & l1 & !l2)*; l0 & l1 & l2>tt")
        reward = 10000

        super().__init__(BreakoutGoalFeatureExtractor(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows),
                         set(lines),
                         f,
                         reward)

    @abstractmethod
    def fromFeaturesToPropositional(self, features, **kwargs):
        """map the matrix bricks status to a propositional formula
        first dimension: columns
        second dimension: row
        """
        matrix = features
        lines_status = np.all(matrix == 0.0, axis=kwargs["axis"])
        result = set()
        sorted_symbols = reversed(self.line_symbols) if kwargs["is_reversed"] else self.line_symbols
        for rs, sym in zip(lines_status, sorted_symbols):
            if rs:
                result.add(sym)

        return frozenset(result)


class BreakoutCompleteRowsTemporalEvaluator(BreakoutCompleteLinesTemporalEvaluator):
    """Temporal evaluator for complete rows in order"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, bottom_up=True):
        super().__init__(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows)
        self.bottom_up = bottom_up

    def fromFeaturesToPropositional(self, features, **kwargs):
        """complete rows from bottom-to-up or top-to-down, depending on self.bottom_up"""
        return super().fromFeaturesToPropositional(features, axis=0, is_reversed=self.bottom_up)


class BreakoutCompleteColumnsTemporalEvaluator(BreakoutCompleteLinesTemporalEvaluator):
    """Temporal evaluator for complete columns in order"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, left_right=True):
        super().__init__(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows)
        self.left_right = left_right

    def fromFeaturesToPropositional(self, features, **kwargs):
        """complete columns from left-to-right or right-to-left, depending on self.left_right"""
        return super().fromFeaturesToPropositional(features, axis=1, is_reversed=not self.left_right)