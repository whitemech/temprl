import numpy as np
from pythogic.base.Formula import AtomicFormula, PathExpressionSequence, PathExpressionStar, \
    PathExpressionEventually, And, Not
from pythogic.base.Alphabet import Alphabet
from pythogic.base.Symbol import Symbol

from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator


class BreakoutBUTemporalEvaluator(TemporalEvaluator):
    """Breakout bottom-up rows deletion temporal evaluator"""

    def __init__(self):
        self.row_symbols = [Symbol(r) for r in ["r0", "r1", "r2"]]
        rows = self.row_symbols
        atoms = [AtomicFormula(r) for r in rows]
        alphabet = Alphabet(set(rows))
        f = PathExpressionEventually(
            PathExpressionSequence.chain([
                PathExpressionStar(And.chain([Not(atoms[0]), Not(atoms[1]), Not(atoms[2])])),
                PathExpressionStar(And.chain([atoms[0], Not(atoms[1]), Not(atoms[2])])),
                PathExpressionStar(And.chain([atoms[0], atoms[1], Not(atoms[2])])),
            ]),
            And.chain([atoms[0], atoms[1], atoms[2]])
        )
        reward = 10000
        super().__init__(alphabet, f, reward)


    def feature_extractor(self, state):
        # TODO: state here is a BreakoutState, but this is wrong! It should extend Space
        return state.bricks.bricks_status_matrix

    def fromFeaturesToPropositional(self, features):
        matrix = features
        row_status = np.all(matrix == 0.0, axis=1)
        result = set()
        for rs, sym in zip(row_status, reversed(self.row_symbols)):
            if rs:
                result.add(sym)

        return frozenset(result)
