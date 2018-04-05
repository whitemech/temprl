from breakout_env.envs.Breakout import Breakout
from breakout_env.wrappers.observation_wrappers import BreakoutFullObservableStateWrapper
from gym.spaces import Discrete

from examples.breakout.breakout_example import BreakoutBUTemporalEvaluator
from rltg.utils.Renderer import Renderer
from rltg.agents.TGAgent import TGAgent
from rltg.brains.TDBrain import Sarsa
from rltg.exploration_policies.RandomPolicyWithDecay import RandomPolicy
from rltg.utils.StatsManager import StatsManager

conf = {
    "observation": "number_discretized",
    "bricks_rows": 3,
    'bricks_color': [200, 180, 160, 140, 120, 100][:3],
    'bricks_reward': [6, 5, 4, 3, 2, 1][-3:],
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
    # env = BreakoutDiscreteStateWrapper(env)
    env = BreakoutFullObservableStateWrapper(env)
    # agent = RLAgent(RandomPolicy(2), Sarsa(Discrete(80*80*105), Discrete(2)))
    agent = TGAgent(RandomPolicy(2), Sarsa(Discrete(80*80*105), Discrete(2)), [BreakoutBUTemporalEvaluator()])
    renderer = Renderer()
    stats = StatsManager()


    # Main training loop
    for ep in range(10000):
        t = 0
        total_reward = 0

        state = env.reset()
        state, reward, done, info = env.step(1)

        while not done:
            action = agent.act(state)
            state2, reward, done, info = env.step(ID2ACTION[action])
            total_reward += reward

            agent.observe(state, action, reward, state2)
            agent.replay()

            state = state2

            if done:
                break

            # renderer.update(env.render())

            t += 1

        agent.reset()
        stats.update(len(agent.brain.Q), total_reward)
        stats.print_summary(ep, t, len(agent.brain.Q), total_reward, agent.exploration_policy.epsilon)



if __name__ == '__main__':
    main()
