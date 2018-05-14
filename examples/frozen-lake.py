import gym

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import QLearning, Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.trainer import Trainer
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
        RandomPolicy(action_space, epsilon=0.1),
        QLearning(observation_space, action_space, gamma=0.9, alpha=0.05, nsteps=1)
    )

    tr = Trainer(env, agent, n_episodes=20000, resume=False,
                 stopping_conditions=(AvgRewardPercentage(window_size=100, target_mean=0.75),))
    tr.main()


