from abc import abstractmethod

import sys
from RLGames.Sapientino import COLORS
from RLGames.gym_wrappers.GymSapientino import GymSapientino
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from flloat.parser.ltlf import LTLfParser
from gym.spaces import Tuple

from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.agents.brains.TDBrain import QLearning, Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.trainer import Trainer
from rltg.utils.Renderer import PygameRenderer


class SapientinoRobotFeatureExtractor(RobotFeatureExtractor):
    pass


class SapientinoNRobotFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["x"],
                input["y"],)

class SapientinoDRobotFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
            obs_space.spaces["theta"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["x"],
                input["y"],
                input["theta"])


class SapientinoTEFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["color"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["color"],)



class SapientinoTemporalEvaluator(TemporalEvaluator):
    """Breakout temporal evaluator for delete columns from left to right"""

    def __init__(self, input_space, gamma=0.99, on_the_fly=False):
        self.color_syms = [Symbol(c) for c in COLORS] + [Symbol("no_color")]
        self.bip = Symbol("bip")

        parser = LDLfParser()

        # the formula
        sb = str(self.bip)
        not_bip = ";(!%s)*;"%sb
        and_bip = lambda x: str(x) + " & " + sb
        # every color-bip in sequence, no bip between colors.
        formula_string = "<(!%s)*;"%sb + not_bip.join(map(and_bip, self.color_syms[:-1])) + ">tt"
        print(formula_string)
        f = parser(formula_string)

        reward = 1

        super().__init__(SapientinoTEFeatureExtractor(input_space),
                         set(self.color_syms).union({self.bip}),
                         f,
                         reward,
                         gamma=gamma,
                         on_the_fly=on_the_fly)


    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        res = set()
        # bip action
        if action == 4:
            res.add(self.bip)

        c = features[0]
        color_sym = self.color_syms[c]
        res.add(color_sym)

        return res


if __name__ == '__main__':
    gamma = 0.99
    on_the_fly = False
    differential = False

    env = GymSapientino(differential=differential)
    if differential:
        feat_ext = SapientinoDRobotFeatureExtractor(env.observation_space)
    else:
        feat_ext = SapientinoNRobotFeatureExtractor(env.observation_space)


    '''Temporal goal - visit all the colors in a given order'''
    agent = TGAgent(feat_ext,
                    RandomPolicy(env.action_space, epsilon=0.1),#, epsilon_start=1.0, decaying_steps=10000),
                    QLearning(None, env.action_space, alpha=0.1, gamma=gamma, nsteps=1),
                    [SapientinoTemporalEvaluator(env.observation_space, gamma=gamma, on_the_fly=on_the_fly)])


    t = Trainer(env, agent,
        n_episodes=100000,
        resume=False,
        eval=False,
        # resume = True,
        # eval = True,
        # renderer=PygameRenderer(delay=0.01)
    )

    t.main()
