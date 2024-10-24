#!/usr/bin/env python3

import numpy as np
load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
play = __import__('4-play').play


np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, _ = train(env, Q)

env.reset()
total_rewards, rendered_outputs = play(env, Q)

print(f'Total Rewards: {total_rewards}')
for output in rendered_outputs:
    print(output)
