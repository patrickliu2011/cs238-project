import numpy as np
import gym.spaces as spaces
import gym
import copy 

import frozenlake_utils as fl

class CustomFrozenLakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    STATE_TYPES = ["embedded_map"]
    OBSCURE_TYPES = [None, "neighbor"]

    def __init__(self, env_kwargs, env_data_kwargs, state_type="embedded_map", obscure_type=None):
        super(CustomFrozenLakeEnv, self).__init__()
        self._env_kwargs = env_kwargs
        self._env_data_kwargs = env_data_kwargs
        self._state_type = state_type
        assert self._state_type in self.STATE_TYPES, "Invalid state type"
        self._obscure_type = obscure_type
        assert self._obscure_type in self.OBSCURE_TYPES, "Invalid obscure type"

        self._env = fl.get_env(**self._env_kwargs)
        self._env_data = fl.get_env_data(self._env, **self._env_data_kwargs)

        self.action_space = spaces.discrete.Discrete(4)
        self._observation_shape = (
            self._env_data["nrows"], 
            self._env_data["ncols"], 
            len(self._env_data["tile_types"]),
        )
        self.observation_space = spaces.box.Box(
            low=0, high=1, shape=self._observation_shape, dtype=np.uint8
        )

        self.done = False
        self._reset_obscuration()

    def _reset_obscuration(self):
        if self._obscure_type is None:
            pass
        elif self._obscure_type == "neighbor":
            self._observed = np.zeros(self._observation_shape[:-1], dtype=bool)
        else:
            raise NotImplementedError("Invalid obscure type")

    def _update_obscuration(self, observation):
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

    def _get_obscured_observation(self, observation):
        if self._obscure_type is None:
            return observation
        elif self._obscure_type == "neighbor":
            return observation * self._observed[..., None]
        else:
            raise NotImplementedError("Invalid obscure type")

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        observation = fl.state_to_observation(observation, self._env_data, mode=self._state_type)
        observation = observation.astype(np.uint8).reshape(self._observation_shape)
        self._update_obscuration(observation)
        observation = self._get_obscured_observation(observation)
        done = terminated or truncated
        self.done = done
        return observation, reward, done, info
    
    def reset(self):
        self._env = fl.get_env(**self._env_kwargs)
        self._env_data = fl.get_env_data(self._env, **self._env_data_kwargs)
        observation, info = self._env.reset()
        observation = fl.state_to_observation(observation, self._env_data, mode=self._state_type)
        observation = observation.astype(np.uint8).reshape(self._observation_shape)
        self._reset_obscuration()
        self._update_obscuration(observation)
        observation = self._get_obscured_observation(observation)
        self.done = False
        return observation  # reward, done, info can't be included
    
    def render(self, mode='human'):
        if not self.done:
            self._env.render()

    def close(self):
        self._env.close()