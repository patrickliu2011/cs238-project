import stable_baselines3 as sb3
import sb3_contrib

ALGOS = {
    "a2c": sb3.A2C,
    "dqn": sb3.DQN,
    "her": sb3.HER,
    "ppo": sb3.PPO,
    "sac": sb3.SAC,
    "td3": sb3.TD3,
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
    "multi": "MultiInputPolicy",
    "mlplstm": "MlpLstmPolicy",
    "cnnlstm": "CnnLstmPolicy",
    "multilstm": "MultiInputLstmPolicy",
}