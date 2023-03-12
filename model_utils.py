"""
Utils for modeling the environment state.
"""

import numpy as np
import typing

import frozenlake_utils as fl

def is_in_bound(coord: typing.Iterable, shape: typing.Iterable):
    return all(0 <= c < s for c, s in zip(coord, shape))

def update_state_counts(
    state_counts: np.ndarray, 
    observation: np.ndarray, 
    env_data: fl.env_dtype, 
    mode: str
) -> np.ndarray:
    """
    Update counts of tile types observed at each location based on observation.

    Parameters:
    -----------
    state_counts: np.ndarray
        Counts of tile types observed at each location. (H, W, tile types)
    observation: np.ndarray
        Observation from agent. Format depends on `mode`.
    env_data: fl.env_dtype
        Environment data.
    mode: str
        Mode of observation. Can be "neighbor", "map", or "embedded_map".
    
    Returns:
    --------
    state_counts: np.ndarray
        Updated counts of tile types observed at each location.
    """
    if mode == "neighbor":
        update = np.zeros_like(state_counts)
        x, y = observation[0:2]
        neighbors = observation[2:].reshape((3, 3, len(env_data["tile_types"]))).argmax(-1)
        for i, j in np.ndindex((3, 3)):
            coord = (int(x + i - 1), int(y + j - 1))
            if not is_in_bound(coord, update.shape): continue
            update[tuple([*coord, neighbors[i, j]])] += 1
    elif mode == "map": 
        obs = observation[2:].reshape(
            tuple([*state_counts.shape, len(env_data["tile_types"])])
        ).argmax(-1)
        update = np.eye(len(env_data["tile_types"]))[obs]
    elif mode == "embedded_map":
        obs = observation[2:].reshape(
            tuple([*state_counts.shape, len(env_data["tile_types"])])
        ).argmax(-1)
        obs[obs == env_data["tile_type_ids"]["S"]] = env_data["tile_type_ids"]["F"]
        update = np.eye(len(env_data["tile_types"]))[obs]
    else:
        raise ValueError("Invalid state mode.")
    return state_counts + update

def state_counts_to_beliefs(
    state_counts: np.ndarray, 
    env_data: fl.env_dtype, 
    unknown_tile=-1
) -> np.ndarray:
    """
    Convert counts of tile types observed at each location to beliefs about 
    tile types at each location.

    Parameters:
    -----------
    state_counts: np.ndarray
        Counts of tile types observed at each location. (H, W, tile types)
    env_data: fl.env_dtype
        Environment data.
    unknown_tile: int
        Tile type to use for unknown tiles. Defaults to -1.
    
    Returns:
    --------
    state_beliefs: np.ndarray
        Most likely tile at each location, or unobserved. (H, W)
    """
    state_beliefs = np.where(
        np.sum(state_counts, axis=-1) == 0, 
        unknown_tile, 
        np.argmax(state_counts, axis=-1)
    )
    return state_beliefs