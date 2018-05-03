from RLGames.gym_wrappers.GymMinecraft import GymMinecraft
from gym.spaces import Tuple

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.trainer import Trainer
from rltg.utils.Renderer import PygameRenderer


class MinecraftRobotFeatureExtractor(RobotFeatureExtractor):
    pass

class MinecraftNRobotFeatureExtractor(MinecraftRobotFeatureExtractor):

    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (ball_x, ball_y, ball_dir, paddle_x)
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
            obs_space.spaces["theta"],
            obs_space.spaces["task_state"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["x"],
                input["y"],
                input["theta"],
                input["task_state"])


if __name__ == '__main__':
    env = GymMinecraft()

    '''Normal task - no temporal goal'''
    agent = RLAgent(MinecraftNRobotFeatureExtractor(env.observation_space),
                    RandomPolicy(env.action_space, epsilon=0.1, epsilon_start=1.0, decaying_steps=1),
                    QLearning(None, env.action_space, alpha=None, gamma=0.9, nsteps=200))


    t = Trainer(env, agent,
        n_episodes=100000,
        resume=False,
        eval=False,
        # resume = True,
        # eval = True,
        # renderer=PygameRenderer(delay=0.01)
    )
    t.main()
