import numpy as np
import gymnasium as gym

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs, info = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, terminated, truncated , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        done = terminated or truncated
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(V, gamma = 1.0):
    """ Extract the policy given a value-function """
    sh = np.prod(env.env.unwrapped.desc.astype(str).shape)
    policy = np.zeros(sh)
    for s in range(sh):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * V[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    sh = np.prod(env.env.unwrapped.desc.astype(str).shape)
    V = np.zeros(sh)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(V)
        for s in range(sh):
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)]
            V[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - V)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
        print(V.reshape(4,4))
    return V

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    sh = np.prod(env.env.unwrapped.desc.astype(str).shape)
    v = np.zeros(sh)
    eps = 1e-10
    i = 0
    while True:
        prev_v = np.copy(v)
        for s in range(sh):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
        i += 1
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    sh = np.prod(env.env.unwrapped.desc.astype(str).shape)
    policy = np.random.choice(env.action_space.n , size=(sh))  # initialize a random policy
    max_iterations = 200000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at at iteration# %d.' %(i+1))
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    env_name  = 'FrozenLake-v1'
    gamma = 0.99
    env = gym.make(env_name, is_slippery=True) #render_mode="human")
    print(env.unwrapped.desc.astype(str))

    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(optimal_v, gamma)
    print("Optimal Policy:\n", policy.reshape(4,4))
    policy_score = evaluate_policy(env, policy, gamma=1, n=1000)
    print('Value Iteration average score = ', policy_score)

    optimal_policy = policy_iteration(env, gamma)
    print("Optimal Policy:\n", optimal_policy.reshape(4,4))
    # env = gym.make(env_name, is_slippery=False, render_mode="human")
    score = evaluate_policy(env, optimal_policy, gamma=1)
    print('Policy iteration average score = ', score)