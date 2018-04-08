from breakout_env.envs.Breakout import Breakout
from breakout_env.wrappers.observation_wrappers import BreakoutFullObservableStateWrapper
from gym.spaces import Discrete

from examples.breakout.breakout_utils import BreakoutRowBottomUpTemporalEvaluator, BreakoutRobotFeatureExtractor
from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.agents.brains.TDBrain import Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.trainer import Trainer
from rltg.utils.Renderer import Renderer

conf = {
    "observation": "number_discretized",
    "bricks_rows": 3,
    'bricks_color': [200, 180, 160, 140, 120, 100][:3],
    'bricks_reward': [6, 5, 4, 3, 2, 1][:3],
    "paddle_speed": 3,
    'paddle_width': 50,
    "ball_speed": [1, 2],
    'max_step': 50000,
    'lifes': 1
}


def normal_goal():
    env = Breakout(conf)
    # env = BreakoutDiscreteStateWrapper(env)
    # env = BreakoutVectorStateWrapper(env)
    env = BreakoutFullObservableStateWrapper(env)

    observation_space = env.observation_space
    action_space = env.action_space
    feat_ext = BreakoutRobotFeatureExtractor()
    feature_space = feat_ext.output_space

    print(observation_space, action_space, feature_space)
    agent = RLAgent(feat_ext,
                    RandomPolicy(action_space, epsilon_start=1.0, epsilon_end=0.001, decaying_steps=3000000),
                    Sarsa(feature_space, action_space, alpha=None, nsteps=100))

    return env, agent

def temporal_goal():
    env = Breakout(conf)
    env = BreakoutFullObservableStateWrapper(env)

    observation_space = env.observation_space
    action_space = env.action_space

    temp_eval = BreakoutRowBottomUpTemporalEvaluator()
    robot_feat_ext = BreakoutRobotFeatureExtractor()
    feature_space = robot_feat_ext.output_space

    print(observation_space, action_space, feature_space)

    agent = TGAgent(robot_feat_ext,
                    RandomPolicy(action_space, epsilon_start=1.0, epsilon_end=0.001, decaying_steps=3000000),
                    Sarsa(feature_space, action_space, alpha=None, nsteps=100),
                    [temp_eval])

    return env, agent



def main():
    # env, agent = normal_goal()
    env, agent = temporal_goal()
    tr = Trainer(
        env, agent,
        n_episodes=12001,
        resume=False,
        # renderer=Renderer(skip_frame=5)
        eval=False,
    )
    tr.main()



if __name__ == '__main__':
    main()
