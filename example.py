import gymnasium as gym
import time

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True,
               render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    env.render()
    time.sleep(0.01)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        time.sleep(1)
        observation, info = env.reset()
env.render()
env.close()