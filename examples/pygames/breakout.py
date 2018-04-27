from RLGames.gym_wrappers.GymBreakout import GymBreakoutN

from examples.pygames.breakout_utils import BreakoutRobotFeatureExtractor, BreakoutCompleteRowsTemporalEvaluator, \
    BreakoutCompleteColumnsTemporalEvaluator
from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.agents.brains.TDBrain import QLearning, Sarsa
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.agents.feature_extraction import IdentityFeatureExtractor
from rltg.trainer import Trainer
from rltg.utils.Renderer import PygameRenderer

if __name__ == '__main__':
    env = GymBreakoutN(brick_cols=3)

    '''Normal task - no temporal goal'''
    # agent = RLAgent(BreakoutRobotFeatureExtractor(env.observation_space),
    #                 RandomPolicy(env.action_space, epsilon_start=1.0, epsilon_end=0.1, decaying_steps=100000),
    #                 Sarsa(None, env.action_space, alpha=None, gamma=0.99, nsteps=100))

    gamma = 0.99
    on_the_fly = False
    '''Temoral goal - specify how and what to complete (columns, rows or both)'''
    agent = TGAgent(BreakoutRobotFeatureExtractor(env.observation_space),
                    RandomPolicy(env.action_space, epsilon_start=1.0, epsilon_end=0.1, decaying_steps=250000),
                    QLearning(None, env.action_space, alpha=None, gamma=gamma, nsteps=100),

                    # Leave one of the following three option to see the differences:
                    # 1) rows
                    # 2) columns
                    # 3) rows and columns

                    # [BreakoutCompleteRowsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, bottom_up=True, gamma=gamma, on_the_fly=on_the_fly)]
                    # [BreakoutCompleteColumnsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, left_right=True, gamma=gamma, on_the_fly=on_the_fly)]

                    [BreakoutCompleteRowsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, bottom_up=True, on_the_fly=on_the_fly),
                    BreakoutCompleteColumnsTemporalEvaluator(env.observation_space, bricks_cols=env.brick_cols, left_right=True, on_the_fly=on_the_fly)]
                    )


    t = Trainer(env, agent,
        n_episodes=10000,
        resume=False,
        eval=False,
        # resume = True,
        # eval = True,
        # renderer=PygameRenderer(delay=0.01)
    )
    t.main()
