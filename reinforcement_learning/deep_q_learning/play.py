#!/usr/bin/env python3
"""
Testing script
"""


from keras import __version__
import tensorflow as tf
tf.keras.__version__ = __version__

import cv2
import numpy as np

from rl.processors import Processor
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers.legacy import Adam
import gymnasium as gym
import matplotlib.pyplot as plt


class AtariProcessor(Processor):
    """Preprocessing Images"""

    def process_observation(self, observation):
        """Preprocess observation"""
        if isinstance(observation, tuple):
            observation = observation[0]
        # Ensure it's a NumPy array
        observation = np.array(observation)
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84))
        return img

    def process_state_batch(self, batch):
        """Rescale the images"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Clip the rewards between -1 and 1"""
        return np.clip(reward, -1., 1.)

class CompatibilityWrapper(gym.Wrapper):
    """
    Compatibility wrapper for gym env to ensure
    compatibility with older versions of gym
    """

    def step(self, action):
        """
        Take a step in the env using the given action

        :param action: action to be taken in env

        :return: tuple containing
            - observation: obs from env after action taken
            - reward: reward obtain after taking the action
            - done: bool indicating whether episode has ended
            - info: additional information from the env
        """
        observation, reward, terminated, truncated, info = (
            self.env.step(action))
        done = terminated or truncated

        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Reset env and return the initial obs

        :param kwargs: additional args

        :return:
            observation: initial obs of the env
        """
        observation, info = self.env.reset(**kwargs)

        return observation

    def render(self, *args, **kwargs):
        # Call the environment's render method without any arguments
        return self.env.render()  # Removes 'mode' argument


def build_model(input_shape, actions):
    """
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    """
    """
    memory = SequentialMemory(limit=100000, window_length=4)
    policy = GreedyQPolicy()
    dqn = DQNAgent(model=model,
                   nb_actions=actions,
                   memory=memory,
                   processor=AtariProcessor(),
                   gamma=.99,
                   policy=policy,
                   train_interval=4)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


def main():
    """
    """
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = CompatibilityWrapper(env)
    observation = env.reset()

    # Visualise one frame
    plt.imshow(observation, cmap='gray')
    plt.title("Initial Observation")
    plt.axis('off')
    plt.show()

    actions = env.action_space.n
    model = build_model(observation.shape, actions)
    model.load_weights('policy.h5')
    dqn = build_agent(model, actions)

    dqn.test(env, nb_episodes=2, visualize=True)
    env.close()


if __name__ == "__main__":
    main()
