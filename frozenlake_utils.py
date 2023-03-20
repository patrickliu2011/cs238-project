import numpy as np
import typing

import gymnasium as gym
from gymnasium.utils import seeding

env_dtype = typing.Dict[str, typing.Any]
tile_dtype = str

def get_env(size: int = 4, 
            show: bool = False, 
            slip: bool = True,
            max_episode_steps: int = 100,
            rewards: typing.Dict[tile_dtype, float] = {},
            p_start=None,
            p_goal=None
            ):
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
    max_episode_steps: int
        Maximum number of steps allowed per episode
    rewards: typing.Dict
        A dictionary of tile types to override with a different reward.
        tile type -> reward value
    p_start:
        Probability distribution of starting position of the agent. If None, 
        will start at (0, 0). If "uniform", will sample from a uniform distribution.
    p_goal:
        Probability distribution of goal position of the agent. If None,
        will start at (size - 1, size - 1). If "uniform", will sample from a uniform distribution.

    Returns:
    --------
    env: gym.Env
        Instance of the environment
    """
    env_kwargs = {
        "desc": generate_random_map(size=size, p_start=p_start, p_goal=p_goal), 
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
                 tile_types: typing.List[tile_dtype] = ["F", "H", "G"], ratio_hide: float = 0.0):
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
    assert len(tile_locations[tile_type_ids["G"]]) == 1, "There should be exactly one goal tile. {}".format(map)

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
        "ratio_hide": ratio_hide,
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

def state_to_observation(state: int, env_data: env_dtype, mode="neighbor"):
    """Convert state (position index) to observation (numpy array) for an agent."""
    position = np.array(pos2coord(state, env_data))
    if mode == "neighbor":
        neighbors = get_tile_neighbors(state, env_data, radius=1)
        neighbors = tile_id_to_onehot(neighbors.flatten(), env_data)
        observation = np.concatenate([position, neighbors.flatten()])
    elif mode == "map":
        neighbors = env_data["map"]
        neighbors = tile_id_to_onehot(neighbors.flatten(), env_data)
        observation = np.concatenate([position, neighbors.flatten()])
    elif mode == "embedded_map":
        neighbors = env_data["map"].copy()
        neighbors[position[0], position[1]] = env_data["tile_type_ids"]["S"]
        neighbors = tile_id_to_onehot(neighbors.flatten(), env_data)
        observation = neighbors.flatten()
    else: 
        assert False, "Unrecognized state to observation conversion mode"
    return observation

def get_overrides(env_data: env_dtype, ratio: float):
    """Get overrides for a random proportion of holes."""
    nholes = env_data["nholes"]
    nholes_to_remove = int(np.random.binomial(nholes, ratio))
    hole_ids = np.random.choice(nholes, nholes_to_remove, replace=False)
    hole_coords = [env_data["tile_locations"][env_data["tile_type_ids"]["H"]][i] for i in hole_ids]
    return {coord: "F" for coord in hole_coords}

def is_valid(board, max_size, start=(0, 0)) -> bool:
    frontier, discovered = [], set()
    if board[start[0], start[1]] != "S": return False
    frontier.append(start)
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(
    size = 8, p = 0.8, seed = None,
    p_start = None, p_goal = None,
):
    """Generates a random valid map (one that has a path from start to goal)
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps
    Returns:
        A random valid map
    """
    valid = False
    board = []

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        
        s_coord = (0, 0)
        if p_start == "uniform":
            i = np.random.choice(np.arange(board.size))
            s_coord = np.unravel_index(i, board.shape)
        elif isinstance(p_start, np.ndarray):
            i = np.random.choice(np.arange(p_start.size), p=p_start.ravel())
            s_coord = np.unravel_index(i, p_start.shape)
        board[s_coord[0]][s_coord[1]] = "S"
        
        g_coord = (size - 1, size - 1)
        if p_goal == "uniform":
            i = np.random.choice(np.arange(board.size))
            g_coord = np.unravel_index(i, board.shape)
        elif isinstance(p_goal, np.ndarray):
            i = np.random.choice(np.arange(p_goal.size), p=p_goal.ravel())
            g_coord = np.unravel_index(i, p_goal.shape)
        if s_coord[0] == g_coord[0] and s_coord[1] == g_coord[1]: continue
        board[g_coord[0]][g_coord[1]] = "G"

        valid = is_valid(board, size, start=s_coord)
    return ["".join(x) for x in board]