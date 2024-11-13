#!/usr/bin/env python3
"""
Training loop for MC policy gradient
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Train the policy using Monte-Carlo policy gradient.

    Parameters:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Returns:
        list: Score values (rewards obtained during each episode)
    """
    # Initialize weights
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        # Reset the environment and get initial state
        state = env.reset()[0]
        episode_gradients = []
        episode_rewards = []
        score = 0
        done = False

        while not done:
            # Get action and gradient based on current policy
            action, grad = policy_gradient(state, weights)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(
                action)

            # Append reward and gradient to the episode history
            episode_rewards.append(reward)
            episode_gradients.append(grad)

            state = next_state
            done = terminated or truncated

        # Store the score for the episode
        score = sum(episode_rewards)
        scores.append(score)

        print(f"Episode: {episode} Score: {score}")

        for i, gradient in enumerate(episode_gradients):
            # Cumulative discounted rewards
            reward = sum([R * gamma ** R for R in episode_rewards[i:]])

            # update the weights
            weights += alpha * gradient * reward

    return scores
