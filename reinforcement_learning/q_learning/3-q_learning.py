#!/usr/bin/env python3
"""
Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Trains the agent using Q-learning in the given FrozenLake environment.

    Parameters:
        env (gym.Env): The FrozenLake environment instance.
        Q (numpy.ndarray): The Q-table initialized to zeros.
        episodes (int, optional): The total number of episodes to train over.
            Default is 5000.
        max_steps (int, optional): The maximum number of steps per episode.
            Default is 100.
        alpha (float, optional): The learning rate. Default is 0.1.
        gamma (float, optional): The discount rate. Default is 0.99.
        epsilon (float, optional): The initial threshold for epsilon-greedy.
            Default is 1.
        min_epsilon (float, optional): The minimum value for epsilon.
            Default is 0.1.
        epsilon_decay (float, optional): The decay rate for epsilon per
            episode. Default is 0.05.

    Returns:
    - :numpy.ndarray: The updated Q-table after training.
    - :list: A list containing the total reward for each episode.
    """
    rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        rewards_current_ep = 0

        for _ in range(max_steps):

            # Choose action based on epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action in the environment
            new_state, reward, done, _, _ = env.step(action)

            # Update the reward to -1 if the agent fell into a hole
            if done and reward == 0:
                reward = -1

            # Update the Q-table for Q(s, a)
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            # Update to the next state, accumulate the reward
            state = new_state
            rewards_current_ep += reward

            if done:
                break

        # Exploration rate decay
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

        # Add this episode's reward to the list of rewards
        rewards.append(rewards_current_ep)

    return Q, rewards
