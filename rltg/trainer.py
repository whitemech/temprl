def main():
    env = None
    agent = None
    num_episodes = 0
    # agent = RLAgent(env, RandomPolicyWithDecay(2), Sarsa(2))

    # Main training loop
    for ep in range(num_episodes):
        t = 0
        total_reward = 0

        state = env.reset()
        state, reward, done, info = env.step(1)

        while not done:
            action = agent.act(state)
            state2, reward, done, info = env.step(action)
            total_reward += reward

            agent.observe(state, action, reward, state2)
            agent.replay()

            state = state2

            if done:
                break

            t += 1

        agent.reset()

if __name__ == '__main__':
    main()