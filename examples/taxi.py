import gym

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import Sarsa
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.agents.policies.EGreedy import EGreedy
from rltg.trainers.GenericTrainer import GenericTrainer
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper
from rltg.utils.StoppingCondition import AvgRewardPercentage

if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    env = GoalEnvWrapper(env, lambda _, reward, done, __: done and reward==20)

    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(
        IdentityFeatureExtractor(observation_space),
        Sarsa(observation_space, action_space, EGreedy(0.1), alpha=0.1, gamma=0.99, lambda_=0.0)
    )

    tr = GenericTrainer(env, agent, n_episodes=10000,
                        resume=False, eval=False,
                        # resume=True, eval=True,
                        stop_conditions=(AvgRewardPercentage(window_size=100, target_mean=9.0),))
    # tr = Trainer(env, agent, n_episodes=10000, resume=True,  eval=True, window_size=1000)
    tr.main()

