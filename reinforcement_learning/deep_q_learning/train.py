#!/usr/bin/env python3
"""
Training script
"""

import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


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


def build_model(input_shape, actions):
    """
    """
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=4,
              activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    """
    """
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=50000,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


def main():
    """
    """
    env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
    env = AtariPreprocessing(env, screen_size=84,
                             grayscale_obs=True,
                             frame_skip=1,
                             noop_max=30)
    # env = FrameStack(env, num_stack=4)
    env = CompatibilityWrapper(env)
    observation = env.reset()

    print(observation.shape)
    # Visualise one frame
    plt.imshow(observation, cmap='gray')
    plt.title("Initial Observation")
    plt.axis('off')
    plt.show()
    exit()


    input_shape = (84, 84, 4)
    actions = env.action_space.n

    model = build_model(input_shape, actions)
    dqn = build_agent(model, actions)

    dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)

    env.close()


if __name__ == "__main__":
    main()
