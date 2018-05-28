import gym

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import QLearning, Sarsa
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.agents.policies.EGreedy import EGreedy
from rltg.trainers.GenericTrainer import GenericTrainer
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper
from rltg.utils.StoppingCondition import AvgRewardPercentage

if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")
    env = GoalEnvWrapper(env, lambda _, reward, done, __: done and reward==1.0)
    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(
        IdentityFeatureExtractor(observation_space),
        QLearning(observation_space, action_space, EGreedy(0.1), gamma=0.99, alpha=0.1)
    )

    tr = GenericTrainer(env, agent, n_episodes=10000,
                        # resume=False, eval=False,
                        resume=True, eval=True,
                        stop_conditions=(AvgRewardPercentage(window_size=100, target_mean=0.8),))
    tr.main()


