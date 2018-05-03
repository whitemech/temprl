import gym

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import Sarsa, QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.trainer import Trainer
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper

if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    env = GoalEnvWrapper(env, lambda _, reward, done, __: done and reward==20)
    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(
        IdentityFeatureExtractor(observation_space),
        RandomPolicy(action_space, epsilon=0.1, epsilon_start=1.0, decaying_steps=5000),
        QLearning(observation_space, action_space, alpha=0.8, nsteps=1, gamma=1.0)
    )

    tr = Trainer(env, agent, n_episodes=10000, resume=False, eval=False, window_size=1000)
    # tr = Trainer(env, agent, n_episodes=10000, resume=True,  eval=True, window_size=1000)
    tr.main()

