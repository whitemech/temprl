from RLGames.gym_wrappers.GymSimpleGrid import GymSimpleGrid
from gym.spaces import Tuple

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import QLearning, Sarsa
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.agents.policies.EGreedy import EGreedy
from rltg.trainers.GenericTrainer import GenericTrainer
from rltg.utils.StoppingCondition import GoalPercentage


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
    env = GymSimpleGrid(rows=20, cols=20)

    agent = RLAgent(
        SimpleGridRFE(env.observation_space),
        Sarsa(None, env.action_space, EGreedy(0.1), alpha=0.1, gamma=0.99, lambda_=0.9)
    )
    tr = GenericTrainer(env, agent, n_episodes=3000,
                        resume=False, eval=False,
                        # resume=True, eval=True,
                        stop_conditions=(GoalPercentage(25, 1.0),)
                        )

    tr.main()
