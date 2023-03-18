import os
import numpy as np
from itertools import count
from argparse import ArgumentParser

import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import sb3_contrib

import frozenlake_utils as fl
import model_utils as mu
from frozenlake_env import CustomFrozenLakeEnv

# TODO: SB3_contrib algos may not run properly with current code
ALGOS = {
    "a2c": stable_baselines3.A2C,
    "dqn": stable_baselines3.DQN,
    "her": stable_baselines3.HER,
    "ppo": stable_baselines3.PPO,
    "sac": stable_baselines3.SAC,
    "td3": stable_baselines3.TD3,
    "ars": sb3_contrib.ARS,
    "maskableppo": sb3_contrib.MaskablePPO,
    "recurrentppo": sb3_contrib.RecurrentPPO, 
    "qrdqn": sb3_contrib.QRDQN,
    "tqc": sb3_contrib.TQC,
    "trpo": sb3_contrib.TRPO,
}

POLICIES = {
    "mlp": "MlpPolicy",
    "cnn": "CnnPolicy",
    "multiinput": "MultiInputPolicy",
}

def main(args):
    from gym.envs.registration import register
    register(
        id="CustomFrozenLake-v1",
        entry_point=CustomFrozenLakeEnv,
        max_episode_steps=args.max_episode_steps,
    )

    from stable_baselines3.common.logger import configure
    log_path = os.path.join(args.log_dir, args.exp_name)
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    env_kwargs = {
        "size": args.size,
        "show": False,
        "slip": args.slip,
        "max_episode_steps": args.max_episode_steps,
        "rewards": args.reward_overrides,
    }
    env_data_kwargs = {
        "tile_types": ["F", "H", "G", "S"]
    }
    if args.ratio_hide > 0:
        env = fl.get_env(**env_kwargs)
        env_data = fl.get_env_data(env, **env_data_kwargs)
        env_data_kwargs["overrides"] = fl.get_overrides(env_data, args.ratio_hide)
    state_type = args.state_type

    custom_env = CustomFrozenLakeEnv(env_kwargs, env_data_kwargs, state_type=state_type)
    check_env(custom_env, warn=True, skip_render_check=True)

    vec_env = make_vec_env(
        "CustomFrozenLake-v1", 
        n_envs=args.num_envs, 
        env_kwargs={"env_kwargs": env_kwargs, "env_data_kwargs": env_data_kwargs, "state_type": state_type}
    )

    model_kwargs = {}
    for var in ["gamma", "batch_size", "device"]:
        if getattr(args, var) is not None:
            model_kwargs[var] = getattr(args, var)
    
    ckpt_path = os.path.join(args.ckpt_dir, args.exp_name)
    algo = ALGOS[args.algo]
    policy = POLICIES[args.policy]
    if args.init_ckpt is not None:
        model = algo.load(args.init_ckpt, env=vec_env)
    else:
        model = algo(policy, vec_env, verbose=1, **model_kwargs)
    model.set_logger(new_logger)
    model.learn(total_timesteps=args.train_timesteps, progress_bar=True)
    model.save(ckpt_path)

    del model

    model = algo.load(ckpt_path)
    env_kwargs_copy = env_kwargs.copy()
    env_kwargs_copy["show"] = True
    vec_env = make_vec_env(
        "CustomFrozenLake-v1", 
        n_envs=1, 
        env_kwargs={"env_kwargs": env_kwargs_copy, "env_data_kwargs": env_data_kwargs, "state_type": state_type}
    )
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
    parser.add_argument("--max-episode-steps", type=int, default=100,
                        help="Maximum number of steps per episode")
    parser.add_argument("--reward-overrides", type=str, nargs="*", default=["H:-1", "F:-0.01", "G:10", "S:-0.01"],
                        help="List of tile types to override rewards for, formatted as \"H:-1 F:-0.01\"")
    parser.add_argument("--ratio-hide", type=float, default=0,
                        help="Ratio of holes to hide")
    parser.add_argument("--state-type", type=str, default="embedded_map",
                        help="Type of state representation to use. Currently only supports \"embedded_map\"")
    
    # Training/eval arguments
    parser.add_argument("--train-timesteps", type=int, default=25_000,
                        help="Number of timesteps to train for.")
    parser.add_argument("--show-episodes", type=int, default=20,
                        help="Number of episodes to display for.")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of environments to run in parallel.")

    # Model arguments
    parser.add_argument("--algo", type=str, default="ppo", choices=ALGOS.keys(),
                        help="SB3 RL algorithm to use (ppo, dqn, etc.)")
    parser.add_argument("--policy", type=str, default="mlp", choices=POLICIES.keys(),
                        help="SB3 policy type (mlp, cnn, etc.)")
    parser.add_argument("--batch-size", type=int, default=None, # 128,
                        help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=None, # 0.99,
                        help="Discount factor for training")
    parser.add_argument("--device", type=str, default=None, # "cuda",
                        help="Device to train on (cpu or cuda)")

    # Saving/logging arguments
    parser.add_argument("--log-dir", type=str, default="sb3_log",
                        help="Directory to save logs to.")
    parser.add_argument("--ckpt-dir", type=str, default="sb3_ckpt",
                        help="Directory to save checkpoints to.")
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="Path to initial checkpoint to load from. If not specified, will start from scratch.")

    args = parser.parse_args()
    reward_overrides = {}
    for kv in args.reward_overrides:
        k, v = kv.split(":")
        reward_overrides[k] = float(v)
    args.reward_overrides = reward_overrides

    print(args)
    main(args)