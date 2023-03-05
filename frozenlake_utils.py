import numpy as np
import typing

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env_dtype = typing.Dict[str, typing.Any]
tile_dtype = str

def get_env(size: int = 4, 
            show: bool = False, 
            slip: bool = True,
            rewards: typing.Dict[tile_dtype, float] = {}):
    """
    Get instance of environment with random hole positions.

    Parameters:
    -----------
    size: int
        Side length of environment
    show: bool
        Create the environment in a render-able format
    slip: bool
        Allow slipping in the environment
    rewards: typing.Dict
        A dictionary of tile types to override with a different reward.
        tile type -> reward value

    Returns:
    --------
    env: gym.Env
        Instance of the environment
    """
    env_kwargs = {
        "desc": generate_random_map(size=size), 
        "is_slippery": slip,
    }
    if show:
        env_kwargs["render_mode"] = "human"
    env = gym.make('FrozenLake-v1', **env_kwargs)
    map = env.unwrapped.desc.astype(str)

    # Update transition rewards
    for pos in env.unwrapped.P:
        for act in env.unwrapped.P[pos]:
            transitions = []
            for prob, next_pos, reward, end in env.unwrapped.P[pos][act]:
                next_tile = map[np.unravel_index(next_pos, map.shape)]
                if next_tile in rewards: 
                    reward = float(rewards[next_tile])
                transitions.append((prob, next_pos, reward, end))
            env.unwrapped.P[pos][act] = transitions

    return env
    

def get_env_data(env, pad_tile: tile_dtype = "B", overrides: typing.Dict = {},
                 tile_types: typing.List[tile_dtype] = ["F", "H", "G"]):
    """
    Get environment data.

    Parameters:
    -----------
    env: gym.Env or np.ndarray
        The environment or map to get data from.
    pad_tile: tile_dtype
        The tile to pad the map with.
    overrides: typing.Dict
        A dictionary of tiles to override with a different tile type.
        tile coordinates -> tile type
    tile_types: typing.List[tile_dtype]
        List of tiles. Order will be preserved in assigning tiles to indices.

    Returns:
    --------
    env_data: env_dtype
        A dictionary of environment data.
    """
    if isinstance(env, np.ndarray):
        map = env
    else:
        map = env.unwrapped.desc.astype(str)

    for coord, tile in overrides.items():
        map[coord] = tile

    nrows, ncols = map.shape
    
    if "S" not in tile_types:
        map[map == "S"] = "F" # Remove start tile
    if pad_tile not in tile_types:
        tile_types.append(pad_tile)
    tile_type_ids = {tp: i for i, tp in enumerate(tile_types)}
    assert len(tile_types) == len(np.unique(tile_types)), "Tile types should be unique."

    map_to_idmap = np.vectorize(lambda x: tile_type_ids[x])
    map = map_to_idmap(map).astype(int)

    tile_locations = {}
    for id in tile_type_ids.values():
        tile_locations[id] = list(zip(*[arr.tolist() for arr in np.where(map == id)]))
    assert len(tile_locations[tile_type_ids["G"]]) == 1, "There should be exactly one goal tile."

    pad_tile_id = tile_type_ids[pad_tile]
    padded_map = np.pad(map, 1, mode="constant", constant_values=pad_tile_id)

    return {
        "nrows": nrows,
        "ncols": ncols,
        "map": map,
        "padded_map": padded_map,
        "pad_tile": pad_tile,
        "pad_tile_id": pad_tile_id,
        "nholes": len(tile_locations[tile_type_ids["H"]]) if "H" in tile_type_ids else 0,
        "tile_locations": tile_locations,
        "tile_types": tile_types,
        "tile_type_ids": tile_type_ids,
    }

def pos2coord(pos: int, env_data: env_dtype):
    """Convert position to coordinates."""
    return np.unravel_index(pos, (env_data["nrows"], env_data["ncols"]))

def coord2pos(coord: typing.Iterable[int], env_data: env_dtype):
    """Convert coordinates to position."""
    return np.ravel_multi_index(coord, (env_data["nrows"], env_data["ncols"]))

def get_tile_type(pos: int, env_data: env_dtype):
    """Get tile type at position."""
    return env_data["map"][pos2coord(pos, env_data)]

def get_tile_neighbors(pos: int, env_data: env_dtype, radius: int = 1):
    """Get neighbors of tile at position."""
    coord = np.array(pos2coord(pos, env_data)) + 1
    dcoord = np.arange(-radius, radius + 1).astype(int)
    ix = np.ix_(*[coord[i] + dcoord for i in range(len(coord))])
    return env_data["padded_map"][ix]

def pos_to_onehot(pos: int, env_data: env_dtype):
    """Convert position to one-hot vector."""
    return np.eye(np.size(env_data["map"]))[pos]

def tile_id_to_onehot(tile_id: int, env_data: env_dtype):
    """Convert tile id to one-hot vector."""
    return np.eye(len(env_data["tile_types"]))[tile_id]

def state_to_observation(state: int, env_data: env_dtype):
    """Convert state (position index) to observation (numpy array) for an agent."""
    position = np.array(pos2coord(state, env_data))
    neighbors = get_tile_neighbors(state, env_data, radius=1)
    neighbors = tile_id_to_onehot(neighbors.flatten(), env_data)
    observation = np.concatenate([position, neighbors.flatten()])
    return observation

def get_overrides(env_data: env_dtype, ratio: float):
    """Get overrides for a random proportion of holes."""
    nholes = env_data["nholes"]
    nholes_to_remove = int(nholes * ratio)
    hole_ids = np.random.choice(nholes, nholes_to_remove, replace=False)
    hole_coords = [env_data["tile_locations"][env_data["tile_type_ids"]["H"]][i] for i in hole_ids]
    return {coord: "F" for coord in hole_coords}