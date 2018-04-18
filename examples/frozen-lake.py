import gym

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import QLearning, Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.trainer import Trainer
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper

if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")
    env = GoalEnvWrapper(env, lambda _, reward, done, __: done and reward==1.0)
    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(
        IdentityFeatureExtractor(observation_space),
        RandomPolicy(action_space, epsilon_end=0.1, decaying_steps=5000),
        Sarsa(observation_space, action_space, alpha=0.1, nsteps=1)
    )

    tr = Trainer(env, agent, n_episodes=20000)
    tr.main()

