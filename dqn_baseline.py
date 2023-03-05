import dqn_utils
import frozenlake_utils
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
steps_done = 0
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# training loop and stuff
def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = dqn_utils.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # print(batch.state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch.type(torch.int64))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def select_action(env, state, policy_net, greedy=False, one_hot=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or greedy:
        with torch.no_grad():
            if one_hot:
                state = F.one_hot(state, num_classes=ob_space).unsqueeze(0).to(dtype=torch.float32, device=device)
            return policy_net(state).argmax().view(1, 1).type(torch.int64)
    else:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.float32)
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.int64)


def train(env, env_data, memory, policy_net, target_net, optimizer, num_episodes=1000):
    episode_durations = []
    for i_episode in range(num_episodes):
        env = frozenlake_utils.get_env(size=4, show=False, slip=False)
        env_data = frozenlake_utils.get_env_data(env)
        state, info = env.reset()
        state = frozenlake_utils.state_to_observation(state, env_data)
        state = torch.as_tensor(state, device=device).float().unsqueeze(0)
        # print(f'state size: {state.shape}')
        # state = torch.as_tensor([state], device=device).unsqueeze(0)
        for t in count():
            action = select_action(env, state, policy_net, one_hot=False)
            # print(f'action: {action}')
            observation, reward, terminated, truncated, _ = env.step(action.item())
            observation = frozenlake_utils.state_to_observation(observation, env_data)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device).unsqueeze(0).float()
                # print(f'next state: {next_state}')
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            if done:
                # episode_durations.append(t + 1)
                episode_durations.append(reward)
                plot_durations(episode_durations)
                break

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def test(policy_net):
    for i_episode in range(50):
        # Initialize the environment and get it's state
        env = frozenlake_utils.get_env(size=4, show=True, slip=False)
        env_data = frozenlake_utils.get_env_data(env)
        state, info = env.reset()
        state = frozenlake_utils.state_to_observation(state, env_data)
        state = torch.as_tensor(state, device=device).float().unsqueeze(0)
        # state = torch.tensor([state], device=device)
        for t in count():
            print(state.shape)
            action = select_action(env, state, policy_net, greedy=True, one_hot=False)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            observation = frozenlake_utils.state_to_observation(observation, env_data)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device).float()
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            if done:
                break

def main():
    print("MAIN")
    env = frozenlake_utils.get_env(size=4, show=False, slip=False)
    env_data = frozenlake_utils.get_env_data(env)
    n_actions = env.action_space.n
    ob_space = env.observation_space.n
    # print(f'original object space: {ob_space}')
    state, info = env.reset()
    obs = frozenlake_utils.state_to_observation(state, env_data)
    # n_observations = ob_space if isinstance(state, int) else len(state)
    # n_observations = ob_space
    n_observations = len(obs)
    memory = dqn_utils.ReplayMemory(10000)
    # print(f'n_observations: {n_observations}')
    policy_net = dqn_utils.DQN(n_observations, n_actions)
    target_net = dqn_utils.DQN(n_observations, n_actions)
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    train(env, env_data, memory, policy_net, target_net, optimizer, 200)
    test(policy_net)

if __name__ == "__main__":
    main()