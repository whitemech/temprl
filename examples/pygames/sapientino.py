from abc import abstractmethod

from RLGames.Sapientino import COLORS
from RLGames.gym_wrappers.GymSapientino import GymSapientino
from flloat.base.Symbol import Symbol
from flloat.parser.ltlf import LTLfParser
from gym.spaces import Tuple

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.trainer import Trainer


class SapientinoRobotFeatureExtractor(RobotFeatureExtractor):
    pass


class SapientinoNRobotFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
            obs_space.spaces["theta"],
            obs_space.spaces["color"],
            obs_space.spaces["RAState"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["x"],
                input["y"],
                input["theta"],
                input["color"],
                int(input["RAState"]))


# class SapientinoTemporalEvaluator(TemporalEvaluator):
#     """Breakout temporal evaluator for delete columns from left to right"""
#
#     def __init__(self, input_space, bricks_cols=3, bricks_rows=3, lines_num=3, gamma=0.99, on_the_fly=False):
#         symbols = {Symbol(c) for c in COLORS}
#         symbols.add(Symbol("bip"))
#
#         parser = LTLfParser()
#         # the formula
#         f = parser(
#             "<(!l0 & !l1 & !l2)*;(l0 & !l1 & !l2);(l0 & !l1 & !l2)*;(l0 & l1 & !l2); (l0 & l1 & !l2)*; l0 & l1 & l2>tt")
#         reward = 10000
#
#         # super().__init__(SapientinoNRobotFeatureExtractor(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows),
#         #                  set(lines),
#         #                  f,
#         #                  reward,
#         #                  gamma=gamma,
#         #                  on_the_fly=on_the_fly)
#
#     @abstractmethod
#     def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
#         pass



if __name__ == '__main__':
    env = GymSapientino()

    '''Normal task - no temporal goal'''
    agent = RLAgent(SapientinoNRobotFeatureExtractor(env.observation_space),
                    RandomPolicy(env.action_space, epsilon=0.1),
                    QLearning(None, env.action_space, alpha=0.1, gamma=0.9, nsteps=100))


    t = Trainer(env, agent,
        n_episodes=100000,
        resume=False,
        eval=False,
        # resume = True,
        # eval = True,
        # renderer=PygameRenderer(delay=0.05)
    )

    t.main()
