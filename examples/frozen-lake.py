import gym

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains import QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.trainer import Trainer
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper

if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")
    env = GoalEnvWrapper(env, lambda _, reward, done, __: done and reward==1.0)
    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(RandomPolicy(action_space, decaying_steps=1000), QLearning(observation_space, action_space, alpha=None, nsteps=10))

    tr = Trainer(env, agent, n_episodes=10000)
    tr.main()

