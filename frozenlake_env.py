import numpy as np
import gymnasium.spaces as spaces
import gymnasium as gym
import copy

import frozenlake_utils as fl
import iter_utils

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
        self._reset_obscuration()
        self._reset_suggestion()

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
        else:
            raise NotImplementedError("Invalid obscure type")

    def _get_obscured_obs_map(self, observation_map):
        if self._obscure_type is None:
            return observation_map
        elif self._obscure_type == "neighbor":
            return observation_map[0] * self._observed[..., None]
        else:
            raise NotImplementedError("Invalid obscure type")

    def _reset_suggestion(self):
        if self._guide_kwargs is None or self._guide_kwargs["type"] is None:
            pass
        elif self._guide_kwargs["type"] == "vi": #value iteration
            optimal_v = iter_utils.value_iteration(self._env, self._guide_kwargs.get("gamma", 0.9))
            self._optimal_policy = iter_utils.extract_policy(self._env, optimal_v)
        else:
            raise NotImplementedError("Invalid guide type")

    def _update_suggestion(self, observation_map):
        if self._guide_kwargs is None or self._guide_kwargs["type"] is None:
            pass
        elif self._guide_kwargs["type"] == "vi":
            self._suggestion = self._optimal_policy[self._env.unwrapped.s]
        else:
            raise NotImplementedError("Invalid guide type")

    def _get_suggestion(self, observation_map):
        if self._guide_kwargs is None or self._guide_kwargs["type"] is None:
            suggestion = np.zeros(self._suggestion_dim, dtype=np.uint8)
            suggestion[0] = 1
            return 0
        elif self._guide_kwargs["type"] == "vi":
            return self._suggestion + 1
        else:
            raise NotImplementedError("Invalid guide type")

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
        return observation, reward, terminated, truncated, info
        # return observation, reward, done, info

    def reset(self):
        self._env = fl.get_env(**self._env_kwargs)
        self._env_data = fl.get_env_data(self._env, **self._env_data_kwargs)
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
        return observation, info  # reward, done, info can't be included

    def render(self, mode='human'):
        if not self.done:
            self._env.render()

    def close(self):
        self._env.close()