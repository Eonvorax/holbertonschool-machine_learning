#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import random

sarsa_lambtha = __import__('2-sarsa_lambtha').sarsa_lambtha


def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


env = gym.make('FrozenLake8x8-v1')
set_seed(env, 0)
Q = np.random.uniform(size=(64, 4))
np.set_printoptions(precision=4)

print(sarsa_lambtha(env, Q, 0.9))
