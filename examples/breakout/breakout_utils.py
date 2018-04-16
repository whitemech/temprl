import numpy as np
from flloat.base.Alphabet import Alphabet
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from gym.spaces import Dict, Discrete, Box, Tuple
from typing import List

from rltg.agents.feature_extraction import FeatureExtractor, TupleFeatureExtractor, RobotFeatureExtractor
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator

# world state space
breakout_obs_space = Dict({
    "paddle_x": Discrete(161),
    "ball_x": Discrete(161),
    "ball_y": Discrete(211),
    "bricks_status_matrix": Box(low=0, high=1, shape=(3, 18), dtype=np.uint8)
})


class BreakoutRobotFeatureExtractor(RobotFeatureExtractor):
    def __init__(self):#, automata_states_len:List[int]=[]):
        """
        Build the Robot Feature Extractor for the Breakout Task.
        As main features, are considered only the x component of the paddle position and the ball coordinates.
        Optionally, can be specified the dimensions of the automaton state space, one for each temporal goal. They will
        be combined with the robot feature space by cartesian product, thanks to the TupleFeatureExtractor.

        :param automata_states_len: a list of integers, representing the number of states for each automaton.
                                    (default: [], i.e. no temporal goal)
        """
        # the input space expected by the feature extractor
        obs_space = breakout_obs_space

        # features considered by the robot in this learning task: (paddle_x, ball_x, ball_y)
        robot_feature_space = (
            obs_space.spaces["paddle_x"],
            obs_space.spaces["ball_x"],
            obs_space.spaces["ball_y"]
        )
        output_space = Tuple(robot_feature_space)

        super().__init__(obs_space, output_space)

    def _extract(self, input, **kwargs):
        tuple_state = (
            input["paddle_x"] // 2,
            input["ball_x"] // 2,
            input["ball_y"] // 2,
        )
        return tuple_state


class BreakoutRowBottomUpGoalFeatureExtractor(FeatureExtractor):
    def __init__(self):

        # the input space expected by the feature extractor
        obs_space = breakout_obs_space

        # the output space is just the matrix representing the bricks status
        output_space = obs_space.spaces["bricks_status_matrix"]

        super().__init__(obs_space, output_space)

    def _extract(self, input, **kwargs):
        return input["bricks_status_matrix"]


class BreakoutRowBottomUpTemporalEvaluator(TemporalEvaluator):
    """Breakout temporal evaluator for delete rows from the bottom to the top"""

    def __init__(self):
        self.row_symbols = [Symbol(r) for r in ["r0", "r1", "r2"]]
        rows = self.row_symbols
        # alphabet = Alphabet(set(rows))
        # f = PathExpressionEventually(
        #     PathExpressionSequence.chain([
        #         PathExpressionStar(And.chain([Not(atoms[0]), Not(atoms[1]), Not(atoms[2])])),
        #         PathExpressionStar(And.chain([atoms[0], Not(atoms[1]), Not(atoms[2])])),
        #         PathExpressionStar(And.chain([atoms[0], atoms[1], Not(atoms[2])])),
        #     ]),
        #     And.chain([atoms[0], atoms[1], atoms[2]])
        # )

        parser = LDLfParser()
        f = parser("<(!r0 & !r1 & !r2)*;(r0 & !r1 & !r2)*;(r0 & r1 & !r2)*; r0 & r1 & r2>tt")
        reward = 10000

        super().__init__(BreakoutRowBottomUpGoalFeatureExtractor(), set(rows), f, reward)


    def fromFeaturesToPropositional(self, features):
        """map the matrix bricks status to a propositional formula"""
        matrix = features
        row_status = np.all(matrix == 0.0, axis=1)
        result = set()
        for rs, sym in zip(row_status, reversed(self.row_symbols)):
            if rs:
                result.add(sym)

        return frozenset(result)
