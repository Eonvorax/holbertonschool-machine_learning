#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake

env = load_frozen_lake()
print(env.unwrapped.desc)
print(len(env.unwrapped.P[0][0]))
print(env.unwrapped.P[0][0])

env = load_frozen_lake(is_slippery=True)
print(env.unwrapped.desc)
print(len(env.unwrapped.P[0][0]))
print(env.unwrapped.P[0][0])

desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.unwrapped.desc)

env = load_frozen_lake(map_name='4x4')
print(env.unwrapped.desc)
