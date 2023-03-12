import sys
sys.path.append('path')

import gym
env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)
env.reset()

while True:
#   env.render()
  move = 1
  action = int(move)
  a, b, c, d, e = env.step(action)
  print(a)
  if c:
    print("Done")
    break
env.render()
#   if done:
#     print(reward)
#     print("Episode finished")
#     env.render()
#     break