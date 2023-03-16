import numpy as np
import gym.spaces as spaces
import gym
import copy 

import frozenlake_utils as fl

class CustomFrozenLakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_kwargs, env_data_kwargs, state_type = "embedded_map"):
        super(CustomFrozenLakeEnv, self).__init__()
        self._env_kwargs = env_kwargs
        self._env_data_kwargs = env_data_kwargs
        self._state_type = state_type
        assert self._state_type in ["embedded_map"], "Invalid state type"

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

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        observation = fl.state_to_observation(observation, self._env_data, mode=self._state_type)
        observation = observation.astype(np.uint8).reshape(self._observation_shape)
        done = terminated or truncated
        self.done = done
        return observation, reward, done, info
    
    def reset(self):
        self._env = fl.get_env(**self._env_kwargs)
        self._env_data = fl.get_env_data(self._env, **self._env_data_kwargs)
        observation, info = self._env.reset()
        observation = fl.state_to_observation(observation, self._env_data, mode=self._state_type)
        observation = observation.astype(np.uint8).reshape(self._observation_shape)
        self.done = False
        return observation  # reward, done, info can't be included
    
    def render(self, mode='human'):
        if not self.done:
            self._env.render()

    def close(self):
        self._env.close()