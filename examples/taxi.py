import gym

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains import Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.trainer import Trainer
from rltg.utils.GoalEnvWrapper import GoalEnvWrapper

if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    env = GoalEnvWrapper(env, lambda _, reward, done, __: done and reward==20)
    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(RandomPolicy(action_space), Sarsa(observation_space, action_space, alpha=0.1, nsteps=1))

    tr = Trainer(env, agent, n_episodes=10000)
    tr.main()

