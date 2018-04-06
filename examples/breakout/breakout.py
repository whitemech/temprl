from breakout_env.envs.Breakout import Breakout
from breakout_env.wrappers.observation_wrappers import BreakoutDiscreteStateWrapper
from gym.spaces import Discrete

from rltg.agents.RLAgent import RLAgent
from rltg.agents.brains.TDBrain import Sarsa, QLearning
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.trainer import Trainer

conf = {
    "observation": "number_discretized",
    "bricks_rows": 6,
    'bricks_color': [200, 180, 160, 140, 120, 100][:],
    'bricks_reward': [6, 5, 4, 3, 2, 1][:],
    "paddle_speed": 3,
    'paddle_width': 50,
    "ball_speed": [1, 2],
    'max_step': 50000,
    'lifes': 1
}

ID2ACTION = {0: 2, 1: 3}
ACTION2ID = {2: 0, 3: 1}

def main():
    env = Breakout(conf)
    env = BreakoutDiscreteStateWrapper(env)
    # env = BreakoutFullObservableStateWrapper(env)
    # agent = RLAgent(RandomPolicy(2), Sarsa(Discrete(80*80*105), Discrete(2)))
    # agent = TGAgent(RandomPolicy(2), Sarsa(Discrete(80 * 80 * 105), Discrete(2)), [BreakoutBUTemporalEvaluator()])

    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)
    agent = RLAgent(IdentityFeatureExtractor(Discrete(80*80*105)),
                    RandomPolicy(action_space, epsilon_start=1.0, epsilon_end=0.001, decaying_steps=4000000),
                    QLearning(Discrete(80*80*105), action_space, alpha=None, nsteps=100))
    tr = Trainer(env, agent, n_episodes=10001, resume=False, render=False)
    tr.main()



if __name__ == '__main__':
    main()
