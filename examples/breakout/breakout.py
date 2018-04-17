from breakout_env.envs.Breakout import Breakout
from breakout_env.wrappers.observation_wrappers import BreakoutFullObservableStateWrapper

from examples.breakout.breakout_utils import BreakoutRowBottomUpTemporalEvaluator, BreakoutRobotFeatureExtractor
from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.agents.brains.TDBrain import Sarsa, QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.trainer import Trainer
from rltg.utils.Renderer import PixelRenderer

conf = {
    "observation": "number_discretized",
    "bricks_rows": 3,
    'bricks_color': [200, 180, 160, 140, 120, 100][:3],
    'bricks_reward': [6, 5, 4, 3, 2, 1][:2:-1],
    "paddle_speed": 3,
    'paddle_width': 50,
    "ball_speed": [1, 2],
    'max_step': 15000,
    'lifes': 1
}


def normal_goal():
    env = Breakout(conf)
    env = BreakoutFullObservableStateWrapper(env)

    # observation_space = env.observation_space
    # action_space = env.action_space
    # feat_ext = BreakoutRobotFeatureExtractor()
    # feature_space = feat_ext.output_space
    # print(observation_space, action_space, feature_space)

    agent = RLAgent(BreakoutRobotFeatureExtractor(),
                    RandomPolicy(env.action_space, epsilon_start=1.0, epsilon_end=0.01, decaying_steps=7500000),
                    Sarsa(None, env.action_space, alpha=None, gamma=0.99, nsteps=100))

    return env, agent


def temporal_goal():
    env = Breakout(conf)
    env = BreakoutFullObservableStateWrapper(env)

    # observation_space = env.observation_space
    # action_space = env.action_space
    # robot_feat_ext = BreakoutRobotFeatureExtractor()
    # feature_space = robot_feat_ext.output_space
    # print(observation_space, action_space, feature_space)

    agent = TGAgent(BreakoutRobotFeatureExtractor(),
                    RandomPolicy(env.action_space, epsilon_start=1.0, epsilon_end=0.01, decaying_steps=7500000),
                    Sarsa(None, env.action_space, alpha=None, gamma=0.99, nsteps=200),
                    [BreakoutRowBottomUpTemporalEvaluator()])

    return env, agent


def main():
    # env, agent = normal_goal()
    env, agent = temporal_goal()
    tr = Trainer(
        env, agent,
        n_episodes=20001,
        resume=True,
        eval=False,
        # renderer=Renderer(skip_frame=5),
    )
    tr.main()



if __name__ == '__main__':
    main()
