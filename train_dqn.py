# https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py
"""
Reinforcement Learning (DQN) Tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
            `Mark Towers <https://github.com/pseudo-rnd-thoughts>`_
"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from frozenlake_utils import *
from argparse import ArgumentParser

def main(args):
    env = get_env(args.size, show=args.show_train, slip=args.slip, rewards=args.reward_overrides)
    env_data = get_env_data(env)

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if gpu is to be used
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    class ReplayMemory(object):

        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    class DQN(nn.Module):

        def __init__(self, n_observations: int, n_actions: int):
            super(DQN, self).__init__()
            print(f'num obs: {n_observations}, num actions: {n_actions}')
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x) -> torch.Tensor:
            # print(f'shape for forward: {x.shape}')
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)

    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    TAU = args.tau
    LR = args.lr

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    ob_space = env.observation_space.n
    # Get the number of state observations
    # reset(seed=42)?
    state, info = env.reset()
    env_data = get_env_data(env)
    state = torch.from_numpy(state_to_observation(state, env_data)).float()
    n_observations = ob_space if isinstance(state, int) else len(state)
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state, greedy=False, one_hot=False):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold or greedy:
            with torch.no_grad():
                if one_hot:
                    state = F.one_hot(torch.tensor(state, device=device), num_classes=ob_space).unsqueeze(0).to(dtype=torch.float32)
                return policy_net(state).argmax().view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    episode_rewards = []

    def plot_rewards(show_result=False):
        plt.figure(1)
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Success')
        plt.plot(rewards_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        batch_next_state = tuple(torch.tensor(s) if s is not None else None for s in batch.next_state)
        batch_state = tuple(torch.tensor(s) if s is not None else None for s in batch.state)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch_next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch_next_state
                                                    if s is not None]).to(dtype=torch.float32)
        # state_batch = torch.cat(batch.state)
        state_batch = torch.cat(batch_state).to(dtype=torch.float32)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    for i_episode in range(args.num_episodes):
        # Initialize the environment and get it's state
        env = get_env(args.size, show=args.show_train, slip=args.slip, rewards=args.reward_overrides)
        state, info = env.reset()
        env_data = get_env_data(env)
        state = torch.from_numpy(state_to_observation(state, env_data)).float().unsqueeze(0)
        for t in count():
            action = select_action(state)
            # print(f'action: {action}')
            observation, reward, terminated, truncated, _ = env.step(action.item())
            observation = torch.from_numpy(state_to_observation(observation, env_data)).float()
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # episode_durations.append(t + 1)
                episode_rewards.append(reward)
                plot_rewards()
                break

    print('Complete')
    plot_rewards(show_result=True)
    plt.ioff()
    plt.show()

    ######################################################################
    # Here is the diagram that illustrates the overall resulting data flow.
    #
    # .. figure:: /_static/img/reinforcement_learning_diagram.jpg
    #
    # Actions are chosen either randomly or based on a policy, getting the next
    # step sample from the gym environment. We record the results in the
    # replay memory and also run optimization step on every iteration.
    # Optimization picks a random batch from the replay memory to do training of the
    # new policy. The "older" target_net is also used in optimization to compute the
    # expected Q values. A soft update of its weights are performed at every step.
    #

    # RENDER THE END
    # env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
    env = get_env(args.size, show=True, slip=args.slip, rewards=args.reward_overrides)

    for i_episode in range(50):
        # Initialize the environment and get it's state
        env = get_env(args.size, show=True, slip=args.slip, rewards=args.reward_overrides)
        state, info = env.reset()
        env_data = get_env_data(env)
        state = torch.from_numpy(state_to_observation(state, env_data)).float().unsqueeze(0)
        for t in count():
            action = select_action(state, greedy=True)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            observation = torch.from_numpy(state_to_observation(observation, env_data)).float()
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if done:
                break

if __name__ == "__main__":
    parser = ArgumentParser(description="Train DQN model on FrozenLake environment.")
    parser.add_argument("--size", type=int, default=4, 
                        help="Size of environment")
    parser.add_argument("--slip", action="store_true", 
                        help="Enable slipping in the environment")
    parser.add_argument("--show-train", action="store_true", 
                        help="Render environment while training")
    parser.add_argument("--num-episodes", type=int, default=600,
                        help="Number of episodes to train for.")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of episodes to eval for.")
    parser.add_argument("--reward-overrides", type=str, nargs="*", default=[],
                        help="List of tile types to override rewards for, formatted as \"H:-1 F:-0.01\"")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for training")
    parser.add_argument("--eps-start", type=float, default=0.9,
                        help="Epsilon-greedy epsilon starting value (proportion of random steps)")
    parser.add_argument("--eps-end", type=float, default=0.05,
                        help="Epsilon-greedy epsilon ending value (proportion of random steps)")
    parser.add_argument("--eps-decay", type=float, default=1000,
                        help="Epsilon-greedy epsilon decay timescale")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network update rate for Q learning")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cpu or cuda)")

    args = parser.parse_args()
    reward_overrides = {}
    for kv in args.reward_overrides:
        k, v = kv.split(":")
        reward_overrides[k] = float(v)
    args.reward_overrides = reward_overrides

    print(args)
    main(args)