import os
import numpy as np
from itertools import count
from argparse import ArgumentParser

import gymnasium as gym
from gym.envs.registration import register

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

import frozenlake_utils as fl
from frozenlake_env import CustomFrozenLakeEnv
from constants import ALGOS, POLICIES

def main(args):
    register(
        id="CustomFrozenLake-v1",
        entry_point=CustomFrozenLakeEnv,
        max_episode_steps=args.max_episode_steps,
    )

    log_path = os.path.join(args.log_dir, args.exp_name)
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    env_kwargs = {
        "size": args.size,
        "show": False,
        "slip": args.slip,
        "max_episode_steps": args.max_episode_steps,
        "rewards": args.reward_overrides,
        "p_start": "uniform" if args.random_start_pos else None,
        "p_goal": "uniform" if args.random_goal_pos else None,
    }
    tile_types = ["F", "H", "G", "S"]
    env_data_kwargs = {
        "tile_types": tile_types,
        "ratio_hide": args.ratio_hide,
    }
    if args.ratio_hide > 0:
        env = fl.get_env(**env_kwargs)
        env_data = fl.get_env_data(env, **env_data_kwargs)
        env_data_kwargs["overrides"] = fl.get_overrides(env_data, args.ratio_hide)
    state_type = args.state_type
    obscure_type = args.obscure_type

    guide_kwargs = {
        "type": args.guide_type,
        "schedule": args.guide_schedule,
        "ckpt": args.guide_ckpt,
    }

    custom_env_kwargs = {
        "env_kwargs": env_kwargs,
        "env_data_kwargs": env_data_kwargs,
        "state_type": state_type,
        "obscure_type": obscure_type,
        "guide_kwargs": guide_kwargs,
    }
    custom_env = CustomFrozenLakeEnv(**custom_env_kwargs)
    check_env(custom_env, warn=True, skip_render_check=True)

    vec_env = make_vec_env(
        "CustomFrozenLake-v1",
        n_envs=args.num_envs,
        env_kwargs=custom_env_kwargs
    )

    model_kwargs = {}
    for var in ["gamma", "batch_size", "device"]:
        if getattr(args, var) is not None:
            model_kwargs[var] = getattr(args, var)
    if args.net_arch is not None:
        model_kwargs["policy_kwargs"] = {"net_arch": args.net_arch}

    ckpt_path = os.path.join(args.ckpt_dir, args.exp_name)
    algo = ALGOS[args.algo]
    policy = POLICIES[args.policy]
    if args.init_ckpt is not None:
        model = algo.load(args.init_ckpt, env=vec_env)
    else:
        model = algo(policy, vec_env, verbose=1, **model_kwargs)
    model.set_logger(new_logger)
    if args.train_timesteps > 0:
        model.learn(total_timesteps=args.train_timesteps, progress_bar=True)
        model.save(ckpt_path)

    if args.eval_episodes > 0:
        rewards, lengths = evaluate_policy(model, vec_env, n_eval_episodes=args.eval_episodes, return_episode_rewards=True)
        print("mean_reward:", np.mean(rewards))
        print("std_reward:", np.std(rewards))

        print("mean_duration:", np.mean(lengths))
        print("std_duration:", np.std(lengths))

        print("success_rate:", np.mean(np.array(rewards) > 0))
        print("fail_rate:", np.mean(np.array(rewards) < -0.1))

        results_path = os.path.join(args.results_dir, args.exp_name) + ".txt"
        os.makedirs(args.results_dir, exist_ok=True)
        with open(results_path, "w") as f:
            print("mean_reward:", np.mean(rewards), file=f)
            print("std_reward:", np.std(rewards), file=f)

            print("mean_duration:", np.mean(lengths), file=f)
            print("std_duration:", np.std(lengths), file=f)

            print("success_rate:", np.mean(np.array(rewards) > 0), file=f)
            print("fail_rate:", np.mean(np.array(rewards) < -0.1), file=f)
    del model

    model = algo.load(ckpt_path)
    custom_env_kwargs["env_kwargs"]["show"] = True
    vec_env = make_vec_env(
        "CustomFrozenLake-v1",
        n_envs=1,
        env_kwargs=custom_env_kwargs,
    )

    if args.policy.endswith("lstm"):
        for i_episode in range(args.show_episodes):
            obs = vec_env.reset()
            lstm_states = None
            episode_starts = np.ones(1, dtype=bool)
            for i in count():
                vec_env.render()
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
                obs, rewards, dones, info = vec_env.step(action)
                episode_starts = dones
                if dones:
                    break
    else:
        for i_episode in range(args.show_episodes):
            obs = vec_env.reset()
            for i in count():
                vec_env.render()
                action, _states = model.predict(obs)
                obs, rewards, dones, info = vec_env.step(action)
                if dones:
                    break


if __name__ == "__main__":
    parser = ArgumentParser(description="Train Stable Baselines3 model on FrozenLake environment.")
    parser.add_argument("--exp-name", type=str, default="ppo_frozenlake",
                        help="Name of experiment.")

    # Environment arguments
    parser.add_argument("--size", type=int, default=4,
                        help="Size of environment")
    parser.add_argument("--slip", action="store_true",
                        help="Enable slipping in the environment")
    parser.add_argument("--random-start-pos", action="store_true",
                        help="Randomize start position")
    parser.add_argument("--random-goal-pos", action="store_true",
                        help="Randomize goal position")
    parser.add_argument("--max-episode-steps", type=int, default=100,
                        help="Maximum number of steps per episode")
    parser.add_argument("--reward-overrides", type=str, nargs="*", default=["H:-1", "F:0", "G:1", "S:0"],
                        help="List of tile types to override rewards for, formatted as \"H:-1 F:-0.01\"")
    parser.add_argument("--ratio-hide", type=float, default=0,
                        help="Ratio of holes to hide")
    parser.add_argument("--state-type", type=str, default="embedded_map",
                        help="Type of state representation to use. Currently only supports \"embedded_map\"")
    parser.add_argument("--obscure-type", type=str, default=None,
                        help="Type of obscuring to use.")

    # Guide arguments
    parser.add_argument("--guide-type", type=str, default=None,
                        help="Type of guide to use. Defaults to None.")
    parser.add_argument("--guide-schedule", type=str, default="always",
                        help="When to receive guide suggestion. Defaults to \"always\".")
    parser.add_argument("--guide-ckpt", type=str, default=None,
                        help="SB3 guide checkpoint path")

    # Training/eval arguments
    parser.add_argument("--train-timesteps", type=int, default=25_000,
                        help="Number of timesteps to train for.")
    parser.add_argument("--eval-episodes", type=int, default=1000,
                        help="Number of episodes to evaluate for.")
    parser.add_argument("--show-episodes", type=int, default=20,
                        help="Number of episodes to display for.")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of environments to run in parallel.")

    # Model arguments
    parser.add_argument("--algo", type=str, default="ppo", choices=ALGOS.keys(),
                        help="SB3 RL algorithm to use (ppo, dqn, etc.)")
    parser.add_argument("--policy", type=str, default="multi", choices=POLICIES.keys(),
                        help="SB3 policy type (mlp, cnn, multi)")
    parser.add_argument("--batch-size", type=int, default=None, # 128,
                        help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=None, # 0.99,
                        help="Discount factor for training")
    parser.add_argument("--device", type=str, default=None, # "cuda",
                        help="Device to train on (cpu or cuda)")
    parser.add_argument("--net-arch", type=int, nargs="+", default=None,
                        help="Hidden layer sizes for MLP policy (e.g. 64 64)")

    # Saving/logging arguments
    parser.add_argument("--log-dir", type=str, default="sb3_log",
                        help="Directory to save logs to.")
    parser.add_argument("--ckpt-dir", type=str, default="sb3_ckpt",
                        help="Directory to save checkpoints to.")
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="Path to initial checkpoint to load from. If not specified, will start from scratch.")
    parser.add_argument("--results-dir", type=str, default="sb3_results",
                        help="Directory to save results to.")

    args = parser.parse_args()
    reward_overrides = {}
    for kv in args.reward_overrides:
        k, v = kv.split(":")
        reward_overrides[k] = float(v)
    args.reward_overrides = reward_overrides

    print(args)
    main(args)