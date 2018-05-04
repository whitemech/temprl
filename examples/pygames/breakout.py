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
from RLGames.gym_wrappers.GymBreakout import GymBreakout
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from gym.spaces import Box, Tuple

from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.agents.brains.TDBrain import QLearning, Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import FeatureExtractor, RobotFeatureExtractor
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.trainer import Trainer
from rltg.utils.Renderer import PygameRenderer
from rltg.utils.StoppingCondition import GoalPercentage, CheckAutomataInFinalState


class BreakoutRobotFeatureExtractor(RobotFeatureExtractor):
    pass

class BreakoutNRobotFeatureExtractor(BreakoutRobotFeatureExtractor):

    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (ball_x, ball_y, ball_dir, paddle_x)
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


class BreakoutSRobotFeatureExtractor(BreakoutRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["diff_paddle_ball"],
        ))
        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return input["diff_paddle_ball"],


class BreakoutGoalFeatureExtractor(FeatureExtractor):
    def __init__(self, obs_space, bricks_rows=3, bricks_cols=3):
        output_space = Box(low=0, high=1, shape=(bricks_cols, bricks_rows), dtype=np.uint8)
        super().__init__(obs_space, output_space)

    def _extract(self, input, **kwargs):
        return input["bricks_matrix"]


class BreakoutCompleteLinesTemporalEvaluator(TemporalEvaluator):
    """Breakout temporal evaluator for delete columns from left to right"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, lines_num=3, gamma=0.99, on_the_fly=False):
        self.line_symbols = [Symbol("l%s" % i) for i in range(lines_num)]
        lines = self.line_symbols

        parser = LDLfParser()
        # the formula
        f = parser("<(!l0 & !l1 & !l2)*;(l0 & !l1 & !l2);(l0 & !l1 & !l2)*;(l0 & l1 & !l2); (l0 & l1 & !l2)*; l0 & l1 & l2>tt")
        reward = 10000

        super().__init__(BreakoutGoalFeatureExtractor(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows),
                         set(lines),
                         f,
                         reward,
                         gamma=gamma,
                         on_the_fly=on_the_fly)

    @abstractmethod
    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
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

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, bottom_up=True, gamma=0.99, on_the_fly=False):
        super().__init__(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows, lines_num=bricks_rows, gamma=gamma, on_the_fly=on_the_fly)
        self.bottom_up = bottom_up

    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        """complete rows from bottom-to-up or top-to-down, depending on self.bottom_up"""
        return super().fromFeaturesToPropositional(features, action, axis=0, is_reversed=self.bottom_up)


class BreakoutCompleteColumnsTemporalEvaluator(BreakoutCompleteLinesTemporalEvaluator):
    """Temporal evaluator for complete columns in order"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, left_right=True, gamma=0.99, on_the_fly=False):
        super().__init__(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows, lines_num=bricks_cols, gamma=gamma, on_the_fly=on_the_fly)
        self.left_right = left_right

    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        """complete columns from left-to-right or right-to-left, depending on self.left_right"""
        return super().fromFeaturesToPropositional(features, action, axis=1, is_reversed=not self.left_right)

if __name__ == '__main__':
    env = GymBreakout(brick_cols=3)

    '''Normal task - no temporal goal'''
    agent = RLAgent(BreakoutNRobotFeatureExtractor(env.observation_space),
                    RandomPolicy(env.action_space, epsilon=0.1),
                    QLearning(None, env.action_space, alpha=None, gamma=1.0, nsteps=100))

    gamma = 0.99
    on_the_fly = False
    '''Temoral goal - specify how and what to complete (columns, rows or both)'''
    agent = TGAgent(BreakoutNRobotFeatureExtractor(env.observation_space),
                    RandomPolicy(env.action_space, epsilon=0.1),
                    QLearning(None, env.action_space, alpha=None, gamma=gamma, nsteps=100),

                    # Leave one of the following three option to see the differences:
                    # 1) rows
                    # 2) columns
                    # 3) rows and columns

                    # 1
                    # [BreakoutCompleteRowsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, bottom_up=True, gamma=gamma, on_the_fly=on_the_fly)]

                    # 2
                    [BreakoutCompleteColumnsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, left_right=True, gamma=gamma, on_the_fly=on_the_fly)]

                    # 3
                    # [BreakoutCompleteRowsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, bottom_up=True, on_the_fly=on_the_fly),
                    # BreakoutCompleteColumnsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, left_right=True, on_the_fly=on_the_fly)]
                    )


    t = Trainer(env, agent,
        n_episodes=10000,
        resume=False,
        eval=False,
        # resume = True,
        # eval = True,
        # renderer=PygameRenderer(delay=0.01)
    )
    t.main()
