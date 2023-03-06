from itertools import product

final_command = 'python3 train_dqn.py '

sizes = {4, 6}
slip = {True, False}
num_episode = {100, 400, 1000} 
eval_episode = {1000}
reward_overrides = {"H:0 F:0 G:1", "H:-1 F:-0.01 G:1", "H:-1 F:0.1 G:1"}
gamma = {0.9, 0.95, 0.99}
ratio_hide = {0.5, 1}
state_type = {"map"}    


config_lists = []

args_names = ["--size", "--slip", "--num-episodes", "eval-episodes", "--reward-overrides", "--gamma", "--ratio-hide", "--state-type"]
for tup in product(sizes, slip, num_episode, eval_episode, reward_overrides, gamma, ratio_hide, state_type):
	configs = []
	for x, y in zip(args_names, tup):
		configs.append(str(x))
		configs.append(str(y))

	config_str = " ".join(configs)
	config_lists.append(final_command + config_str)

print(config_lists) 

commands = config_lists

from subprocess import Popen

# run in parallel
processes = [Popen(cmd, shell=True) for cmd in commands]
for p in processes: 
	p.wait()
