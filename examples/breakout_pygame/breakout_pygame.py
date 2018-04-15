from gym.spaces import Discrete, Tuple

from examples.breakout_pygame.Breakout import BreakoutN
from rltg.agents.RLAgent import RLAgent
from rltg.agents.feature_extraction import IdentityFeatureExtractor, TupleFeatureExtractor
from rltg.trainer import Trainer
from rltg.agents.brains.TDBrain import Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy


def main():
    env = BreakoutN()
    env.init(None)

    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    feat_ext = TupleFeatureExtractor(observation_space)
    agent = RLAgent(feat_ext,
                    RandomPolicy(action_space, epsilon_end=0.001, decaying_steps=30000),
                    Sarsa(feat_ext.output_space, action_space, alpha=None, gamma=0.995, nsteps=100)
            )
    tr = Trainer(env, agent, n_episodes=10000)
    tr.main()



if __name__ == '__main__':
    main()
