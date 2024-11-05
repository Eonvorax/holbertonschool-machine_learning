#!/usr/bin/env python3
"""
Monte Carlo algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm for estimating the value function.

    Parameters:
        env: Environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimates.
        policy: Function that takes a state and returns the next action
            to take.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        Updated value estimates V.
    """
    for episode in range(episodes):
        # reset the environment and get initial state
        state = env.reset()[0]
        episode_data = []

        for step in range(max_steps):
            # select action based on policy
            action = policy(state)

            # take action
            next_state, reward, terminated, truncated, _ = env.step(
                action)

            # Append state and reward to the episode history
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            # move to the next state
            state = next_state

        G = 0
        episode_data = np.array(episode_data, dtype=int)

        # Compute the returns for each state in the episode
        for state, reward in reversed(episode_data):
            # calculate this episode's return
            G = reward + gamma * G

            # if this is a novel state
            if state not in episode_data[:episode, 0]:
                # Update the value function V(s)
                V[state] = V[state] + alpha * (G - V[state])

    return V
