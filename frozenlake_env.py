import numpy as np
import gym.spaces as spaces
import gym
import copy

import frozenlake_utils as fl
import iter_utils

from constants import ALGOS, POLICIES

class CustomFrozenLakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    STATE_TYPES = ["embedded_map"]
    OBSCURE_TYPES = [None, "neighbor"]

    def __init__(self, env_kwargs, env_data_kwargs,
                 state_type="embedded_map", obscure_type=None,
                 guide_kwargs=None):
        super(CustomFrozenLakeEnv, self).__init__()
        self._env_kwargs = env_kwargs
        self._env_data_kwargs = env_data_kwargs
        if "overrides" in self._env_data_kwargs:
            del self._env_data_kwargs["overrides"]
        self._guide_kwargs = guide_kwargs
        self._state_type = state_type
        assert self._state_type in self.STATE_TYPES, "Invalid state type"
        self._obscure_type = obscure_type
        assert self._obscure_type in self.OBSCURE_TYPES, "Invalid obscure type"

        self._env = fl.get_env(**self._env_kwargs)
        self._env_data = fl.get_env_data(self._env, **self._env_data_kwargs)

        self.action_space = spaces.discrete.Discrete(4)
        self._obs_map_shape = (
            self._env_data["nrows"],
            self._env_data["ncols"],
            len(self._env_data["tile_types"]),
        )
        self._suggestion_dim = 5

        self.observation_space = spaces.dict.Dict({
            "map": spaces.box.Box(low=0, high=1, shape=self._obs_map_shape, dtype=np.uint8),
            "suggestion": spaces.discrete.Discrete(self._suggestion_dim),
        })

        self.done = False
        if self._guide_kwargs["type"] in ALGOS:
            algo = ALGOS[self._guide_kwargs["type"]]
            self._guide = algo.load(self._guide_kwargs["ckpt"])

    def _reset_obscuration(self):
        if self._obscure_type is None:
            pass
        elif self._obscure_type == "neighbor":
            self._observed = np.zeros(self._obs_map_shape[:-1], dtype=bool)
        else:
            raise NotImplementedError("Invalid obscure type")

    def _update_obscuration(self, observation_map):
        if self._obscure_type is None:
            pass
        elif self._obscure_type == "neighbor":
            x, y = fl.pos2coord(self._env.unwrapped.s, self._env_data)
            for i, j in np.ndindex((3, 3)):
                coord = (int(x + i - 1), int(y + j - 1))
                if not all(0 <= c < s for c, s in zip(coord, self._observed.shape)):
                    continue
                self._observed[coord] = True

    def _get_obscured_obs_map(self, observation_map):
        if self._obscure_type is None:
            return observation_map
        elif self._obscure_type == "neighbor":
            return observation_map * self._observed[..., None]

    def _reset_suggestion(self):
        if self._guide_kwargs is None or self._guide_kwargs["type"] is None:
            return
        elif self._guide_kwargs["type"] == "vi": #value iteration
            optimal_v = iter_utils.value_iteration(self._env, self._guide_kwargs.get("gamma", 0.9))
            self._optimal_policy = iter_utils.extract_policy(self._env, optimal_v)
            # print("V:", optimal_v.reshape((self._env_data["nrows"], self._env_data["ncols"], -1)))
            # print("P:", self._optimal_policy.reshape((self._env_data["nrows"], self._env_data["ncols"], -1)))
        elif self._guide_kwargs["type"] in ALGOS:
            pass
        else:
            raise NotImplementedError("Invalid guide type")
        
        if self._guide_kwargs["schedule"] in ["always", "never", "random"]:
            pass
        elif self._guide_kwargs["schedule"] == "time":
            self._t = 0
        elif self._guide_kwargs["schedule"] in ["hole", "hidden_hole"]:
            self._hole_neighbors = set()
            if self._guide_kwargs["schedule"] == "hole":
                hole_id = self._env_data["tile_type_ids"]["H"]
                hole_locs = self._env_data["tile_locations"][hole_id] if self._env_data["nholes"] > 0 else []
            else:
                assert self._env_data_kwargs.get("ratio_hide", 0) > 0, "hidden_hole schedule requires ratio_hide>0"
                hole_locs = self._overrides.keys()
            for x, y in hole_locs:
                for i, j in np.ndindex((3, 3)):
                    coord = (int(x + i - 1), int(y + j - 1))
                    if not all(0 <= c < s for c, s in zip(coord, self._env_data["map"].shape)):
                        continue
                    self._hole_neighbors.add(fl.coord2pos(coord, self._env_data))
        else:
            raise NotImplementedError("Invalid guide schedule type")

    def _update_suggestion(self, observation_map):
        if self._guide_kwargs is None or self._guide_kwargs["type"] is None:
            return
        elif self._guide_kwargs["type"] == "vi":
            pass
        elif self._guide_kwargs["type"] in ALGOS:
            pass
        
        if self._guide_kwargs["schedule"] in ["always", "never", "random"]:
            pass
        elif self._guide_kwargs["schedule"] == "time":
            self._t += 1

    def _get_suggestion(self, observation_map):
        use_suggestion = False
        if self._guide_kwargs["schedule"] == "always":
            use_suggestion = True
        elif self._guide_kwargs["schedule"] == "random":
            use_suggestion = np.random.choice([True, False])
        elif self._guide_kwargs["schedule"] == "never":
            use_suggestion = False
        elif self._guide_kwargs["schedule"] == "time":
            use_suggestion = (self._t > 30)
        elif self._guide_kwargs["schedule"] in ["hole", "hidden_hole"]:
            use_suggestion = self._env.unwrapped.s in self._hole_neighbors
        
        suggestion = 0
        if not use_suggestion:
            pass
        elif self._guide_kwargs is None or self._guide_kwargs["type"] is None:
            pass
        elif self._guide_kwargs["type"] == "vi":
            suggestion = self._optimal_policy[self._env.unwrapped.s] + 1
        elif self._guide_kwargs["type"] in ALGOS:
            if self._env_data_kwargs.get("ratio_hide", 0) > 0:
                observation_map_copy = observation_map.copy()
                for coord, tile in self._overrides.items():
                    tile_id = self._env_data["tile_type_ids"][tile]
                    if self._obscure_type != "neighbor" or self._observed[coord]:
                        observation_map_copy[coord] = tile_id
                action, _states = self._guide.predict(observation_map_copy)
            else:
                action, _states = self._guide.predict(observation_map)
            suggestion = action + 1
        return int(suggestion)

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        observation = fl.state_to_observation(observation, self._env_data, mode=self._state_type)
        obs_map = observation.astype(np.uint8).reshape(self._obs_map_shape)

        self._update_obscuration(obs_map)
        obs_map = self._get_obscured_obs_map(obs_map)

        self._update_suggestion(obs_map)
        suggestion = self._get_suggestion(obs_map)

        observation = {"map": obs_map, "suggestion": suggestion}
        done = terminated or truncated
        self.done = done
        # return observation, reward, terminated, truncated info
        return observation, reward, done, info

    def reset(self):
        self._env = fl.get_env(**self._env_kwargs)
        self._env_data = fl.get_env_data(self._env, **self._env_data_kwargs)
        if self._env_data_kwargs.get("ratio_hide", 0) > 0:
            self._overrides = fl.get_overrides(self._env_data, self._env_data_kwargs["ratio_hide"])
            self._env_data = fl.get_env_data(self._env, **self._env_data_kwargs, overrides=self._overrides)

        observation, info = self._env.reset()
        obs_map = fl.state_to_observation(observation, self._env_data, mode=self._state_type)
        obs_map = obs_map.astype(np.uint8).reshape(self._obs_map_shape)

        self._reset_obscuration()
        self._update_obscuration(obs_map)
        obs_map = self._get_obscured_obs_map(obs_map)

        self._reset_suggestion()
        self._update_suggestion(obs_map)
        suggestion = self._get_suggestion(obs_map)

        observation = {"map": obs_map, "suggestion": suggestion}
        self.done = False
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        if not self.done:
            self._env.render()

    def close(self):
        self._env.close()