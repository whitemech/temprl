from RLGames.gym_wrappers.GymSapientino import GymSapientino
from gym.spaces import Tuple

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import Sarsa, QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.trainer import Trainer
from rltg.utils.Renderer import PygameRenderer


class SapientinoRobotFeatureExtractor(RobotFeatureExtractor):
    pass

class SapientinoNRobotFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (ball_x, ball_y, ball_dir, paddle_x)
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
