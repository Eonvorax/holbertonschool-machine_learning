#!/usr/bin/env python3
"""
Play the game by exploiting the Q-table
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Simulates an episode of the environment using a trained Q-table by always
    exploiting the Q-values.

    Parameters:
        env (gym.Env): The FrozenLake environment instance.
        Q (numpy.ndarray): The trained Q-table.
        max_steps (int, optional): The maximum number of steps in the episode.
            Default is 100.

    Returns:
        :float: The total rewards accumulated during the episode.
        :list: A list of strings representing the rendered outputs of the
            board state at each step.
    """
    # Reset the environment to the starting state
    state = env.reset()[0]

    done = False
    total_rewards = 0
    rendered_outputs = []

    for _ in range(max_steps):
        # Render the current state of the environment and store the output
        rendered_outputs.append(env.render())

        # Choose action maximizing the Q-value (exploit)
        action = np.argmax(Q[state])

        # Take the action in the environment
        next_state, reward, done, _, _ = env.step(action)

        # update reward
        total_rewards += reward

        # Update to the next state
        state = next_state

        if done:
            break

    # Include the final state of the environment
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
