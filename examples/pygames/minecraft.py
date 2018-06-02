from RLGames.Minecraft import LOCATIONS, TASKS
from RLGames.gym_wrappers.GymMinecraft import GymMinecraft
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from gym.spaces import Tuple

from rltg.agents.TGAgent import TGAgent
from rltg.agents.brains.TDBrain import Sarsa
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.trainers.TGTrainer import TGTrainer


class MinecraftRobotFeatureExtractor(RobotFeatureExtractor):
    pass

class MinecraftNRobotFeatureExtractor(MinecraftRobotFeatureExtractor):

    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (ball_x, ball_y, ball_dir, paddle_x)
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self,   input, **kwargs):
        return (input["x"],
                input["y"])


class MinecraftTEFeatureExtractor(MinecraftRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["location"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["location"],)

class MinecraftTemporalEvaluator(TemporalEvaluator):
    def __init__(self, input_space, formula_string, gamma=0.99, on_the_fly=False):
        self.location_syms = [Symbol(l[0]) for l in LOCATIONS]
        self.get, self.use = Symbol("get"), Symbol("use")

        parser = LDLfParser()
        print(formula_string)
        f = parser(formula_string)
        reward = 1

        super().__init__(MinecraftTEFeatureExtractor(input_space),
                         set(self.location_syms).union({self.get, self.use}),
                         f,
                         reward,
                         gamma=gamma,
                         on_the_fly=on_the_fly)

    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        res = set()

        if action == 4:
            res.add(self.get)
        elif action == 5:
            res.add(self.use)

        l = features[0]
        if l < len(self.location_syms):
            location_sym = self.location_syms[l]
            res.add(location_sym)

        return res

class MinecraftSafetyTemporalEvaluator(MinecraftTemporalEvaluator):
    def __init__(self, input_space, gamma=0.99, on_the_fly=False):

        # the formula
        self.location_syms = [Symbol(l[0]) for l in LOCATIONS]
        ngnu = "(<(get | use) -> (%s) >tt | [true]ff)" % " | ".join(map(str,self.location_syms))
        formula_string = "[true*]" + ngnu

        super().__init__(input_space,
                         formula_string,
                         gamma=gamma,
                         on_the_fly=on_the_fly)

class MinecraftTaskTemporalEvaluator(MinecraftTemporalEvaluator):
    """Breakout temporal evaluator for delete columns from left to right"""

    def __init__(self, input_space, task, gamma=0.99, on_the_fly=False):
        """:param task: a list of subgoals of the following form:
        ['action_location', 'action_location'...]
        e.g.
        ['get_wood', 'use_toolshed', 'get_grass', 'use_workbench']"""


        # the formula
        ngnu = "true*"
        split_task = lambda x: " & ".join(x.split("_"))
        formula_string = "<true*>(<%s;" % ngnu + (";" + ngnu + ";").join(map(split_task, task)) + ">tt)"

        super().__init__(input_space,
                         formula_string,
                         gamma=gamma,
                         on_the_fly=on_the_fly)


    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        res = set()

        if action == 4:
            res.add(self.get)
        elif action == 5:
            res.add(self.use)

        l = features[0]
        if l < len(self.location_syms):
            location_sym = self.location_syms[l]
            res.add(location_sym)

        return res

def temporal_evaluators_from_task(tasks, gamma=0.99, on_the_fly=False):
    res = [MinecraftTaskTemporalEvaluator(env.observation_space, t, gamma=gamma, on_the_fly=on_the_fly) for k, t in list(tasks.items())[:] if "make" in k]
    res.append((MinecraftSafetyTemporalEvaluator(env.observation_space, gamma=gamma, on_the_fly=on_the_fly)))
    return res

if __name__ == '__main__':
    env = GymMinecraft()

    on_the_fly = False
    reward_shaping = True
    gamma = 0.99
    '''Temporal goal - complete every task'''
    agent = TGAgent(MinecraftNRobotFeatureExtractor(env.observation_space),
                    Sarsa(None, env.action_space, alpha=0.1, gamma=gamma, lambda_=0.3),
                    temporal_evaluators_from_task(TASKS, gamma=gamma, on_the_fly=on_the_fly),
                    reward_shaping=reward_shaping
                    )


    t = TGTrainer(env, agent,
        n_episodes=100000,
        resume=False,
        eval=False,
        # resume = True,
        # eval = True,
        # renderer=PygameRenderer(delay=0.01)
    )
    t.main()
