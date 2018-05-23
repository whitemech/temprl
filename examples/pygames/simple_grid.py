from RLGames.gym_wrappers.GymSimpleGrid import GymSimpleGrid
from gym.spaces import Tuple

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.trainer import Trainer
from rltg.utils.Renderer import PygameRenderer


class SimpleGridRFE(RobotFeatureExtractor):
    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (ball_x, ball_y, ball_dir, paddle_x)
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["x"],
                input["y"])


if __name__ == '__main__':
    env = GymSimpleGrid(rows=3, cols=3)

    '''Normal task - no temporal goal'''
    agent = RLAgent(SimpleGridRFE(env.observation_space),
                    RandomPolicy(env.action_space, epsilon=0.1),
                    QLearning(None, env.action_space, alpha=0.1, gamma=1.0, nsteps=1))

    t = Trainer(env, agent,
        n_episodes=1000,
        resume=False,
        eval=False,
        # resume = True,
        # eval = True,
        # renderer=PygameRenderer(delay=0.1)
    )
    t.main()
