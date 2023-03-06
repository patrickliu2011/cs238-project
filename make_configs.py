from itertools import product

final_command = 'python3 train_dqn.py '

sizes = {5}
slip = {True, False}
num_episode = {1000}
eval_episode = {100}
reward_overrides = {"H:0 F:0 G:1", "H:-1 F:-0.01 G:1", "H:-1 F:0.1 G:1"}
gamma = {0.99}
ratio_hide = {0.5, 1}
state_type = {"map"}

additional_commands = " --show-episodes 0 --show-interval -1 --suppress-figs"

config_lists = []

args_names = ["--size", "--slip", "--num-episodes", "--eval-episodes", "--reward-overrides", "--gamma", "--ratio-hide", "--state-type"]
for tup in product(sizes, slip, num_episode, eval_episode, reward_overrides, gamma, ratio_hide, state_type):
	configs = []
	for x, y in zip(args_names, tup):
		if isinstance(y, bool):
			if y:
				configs.append(str(x))
			continue
		configs.append(str(x))
		configs.append(str(y))

	config_str = " ".join(configs)
	exp_name = config_str.replace(" ", "_").replace("--", "")
	config_str += " --exp-name " + exp_name
	config_lists.append(final_command + config_str + additional_commands)

print(config_lists)

commands = config_lists

import subprocess
for cmd in commands:
    subprocess.run(cmd.split(" "))

"""
from subprocess import Popen
from concurrent.futures import ThreadPoolExecutor

max_processes = 5
processes = []

def run_command(command):
    process = Popen(command, shell=True)
    process.wait()
    return process

# run in parallel
with ThreadPoolExecutor(max_workers=max_processes) as executor:
    for command in commands:
        processes.append(executor.submit(run_command, command))
"""
