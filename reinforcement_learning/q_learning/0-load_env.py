#!/usr/bin/env python3
"""
Load FrozenLakeEnv environment
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from the gymnasium library.

    Parameters:
        desc (list of lists, optional):
            A custom description of the map to load.
            Each list represents a row on the map, with the characters:
            - 'S': Start position
            - 'F': Frozen lake (safe)
            - 'H': Hole (dangerous)
            - 'G': Goal
            If desc is provided, map_name will be ignored.

        map_name (str, optional): The name of a pre-defined map to load.
            Supported options include '4x4', '8x8', etc.
            If both desc and map_name are None, a random 8x8 map will be
            generated.

        is_slippery (bool, optional): If True, the ice is slippery
            (stochastic movements). If False, the agent will move
            deterministically.

    Returns:
        gym.Env: The initialized FrozenLake environment.
    """
    return gym.make('FrozenLake-v1', desc=desc,
                    map_name=map_name, is_slippery=is_slippery)
