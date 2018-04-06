from gym.spaces import Discrete

from examples.breakout_iocchi.Breakout import Breakout
from rltg.agents.RLAgent import RLAgent
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.trainer import Trainer
from rltg.agents.brains import Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy


def main():
    env = Breakout()
    # env = BreakoutFullObservableStateWrapper(env)
    # agent = RLAgent(RandomPolicy(2), Sarsa(Discrete(80*80*105), Discrete(2)))
    # agent = TGAgent(RandomPolicy(2), Sarsa(Discrete(80 * 80 * 105), Discrete(2)), [BreakoutBUTemporalEvaluator()])

    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(IdentityFeatureExtractor(), RandomPolicy(action_space), Sarsa(Discrete(80*80*105), action_space, alpha=None, nsteps=200))
    tr = Trainer(env, agent, n_episodes=10000)
    tr.main()



if __name__ == '__main__':
    main()
