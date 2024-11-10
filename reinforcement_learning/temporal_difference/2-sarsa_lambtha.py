#!/usr/bin/env python3
"""
SARSA(λ) algorithm (with eligibility traces)
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy policy.

    Parameters:
        Q (numpy.ndarray): The Q-table, where each entry Q[s, a] represents
            the expected reward for state `s` and action `a`.
        state (int): The current state.
        epsilon (float): The epsilon value for the epsilon-greedy policy.
            With probability `epsilon` the action is chosen randomly
            (explore) and with probability `(1 - epsilon)` the action with the
            highest Q-value is chosen (exploit).

    Returns:
        int: The index of the action to take next.
    """
    # With a sampled random value between 0 and 1:
    if np.random.uniform(0, 1) > epsilon:
        # Exploit: choose the action with the highest Q-value
        return np.argmax(Q[state, :])
    else:
        # Explore: choose a random action
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm (with eligibility traces) to estimate
    a Q-table.

    Parameters:
        env (gym.Env): The FrozenLake environment instance.
        Q (numpy.ndarray): The given Q-table.
        lambtha (float): The eligibility trace factor.
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
        numpy.ndarray: The updated Q-table after training.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        # Reset and choose first action
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        # Init. eligibility traces to zero, for all states
        eligibility_traces = np.zeros_like(Q)

        for steps in range(max_steps):
            # Take the action in the environment
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action based on epsilon-greedy policy
            new_action = epsilon_greedy(Q, new_state, epsilon)

            # TD Error (δ): reward + gamma * V(next_state) - V(state)
            delta = (reward + (gamma * Q[new_state, new_action]) -
                     Q[state, action])

            # Update eligibility traces, apply lambtha decay
            eligibility_traces[state, action] += 1
            eligibility_traces *= lambtha * gamma

            # Update the Q-table
            Q += alpha * delta * eligibility_traces

            # Update to the next state & action
            state = new_state
            action = new_action

            if terminated or truncated:
                break

        # Exploration rate decay
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
