import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
import seaborn as sns
from typing import Tuple, Dict, List
from numpy.typing import NDArray
from tqdm import tqdm
from gym import spaces
import sklearn
from collections import defaultdict
import os
import inspect
import sys
import copy
from Dam_env import *
from DeepQAgent import *
from TestEnv import *




def run_ddq():
    # Load data
    df = pd.read_csv("train_transformed.csv", index_col=0)
    df_val = pd.read_csv("validate_transformed.csv", index_col=0)

    # Set fix parameters
    runs = 3
    n_discrete_actions = 3
    state_space = [10, 4]
    num_episodes = 2
    reward_shaping = False
    reward_shaping_type = 1

    dict_run = {
        "Run": [run_list_item for run_list in [[run] * 2 * num_episodes for run in range(0, runs)] for run_list_item in
                run_list],
        "Episode": list(range(0, num_episodes)) * 2 * runs,
        "Return": [],
        "Stage": list(["Training"] * num_episodes + ["Validation"] * num_episodes) * runs}

    # Define parameters
    gamma = 0.98
    alpha = 0.00005
    end_epsilon = 0.05
    target_update = 5000


    for _ in range(runs):
        print(40 * "___")
        print(f"Run: {_}")
        print("Gamma, Alpha, End Epsilon, Target update freq =", gamma, alpha, end_epsilon, target_update)

        # Build training and validation environment
        env = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=df,
                     warm_start=False, warm_start_step=400, shaping=reward_shaping, shaping_type=reward_shaping_type)

        env_val = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=df_val,
                         warm_start=False, warm_start_step=400)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build agent
        agent = DDQNAgent(env=env, Qnet=QNetwork, device=device, epsilon_greedy=EpsilonGreedyPolicy(),
                          epsilon_start=1.0, epsilon_end=end_epsilon, discount_rate=gamma, alpha=alpha,
                          Buffer=ExperienceReplay, buffer_capacity=len(df) * 4)

        # Run  training episodes and validation
        train_viz_data, val_viz_data, policy, learning_curves, returns, returns_val = agent.run_episodes(
            num_episodes=num_episodes, validation_env=env_val, batch_size=32,
            target_update_freq=target_update)

        concat = returns + returns_val
        dict_run["Return"] += concat

    # save network
    torch.save(agent.online_network.state_dict(), "online_network.pt")

    # save data
    #print(dict_run)
    df = pd.DataFrame(data=dict_run)
    df.to_csv(f"runs/ddq_rewardshaping_{reward_shaping}_type_{reward_shaping_type}.csv")

    df_val = pd.DataFrame(data=val_viz_data)
    df_val.to_csv(f"runs/ddq_valdata_rewardshaping_{reward_shaping}_type_{reward_shaping_type}.csv")



run_ddq()