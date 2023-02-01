import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from data_transformation import DataTransformation
from baselines import *
from thresholds import *
from Damn_env import *
from TabQAgent import *
from Tab_EpsilonGreedy import *

# TODO Make Argparser
# TODO insert Vincents test env



def main():

    # Data transformation
    data_transformer = DataTransformation()

    train_path = "train.xlsx"
    val_path = "validate.xlsx"

    data_transformer.transform(dataframe_path=train_path)
    data_transformer.transform(dataframe_path=val_path)

    # read in files
    train_df = pd.read_csv(train_path.replace(".xlsx", "") + "_transformed.csv", index_col=0)
    val_df = pd.read_csv(val_path.replace(".xlsx", "") + "_transformed.csv", index_col=0)

    # set thresholds for baseline and tabular Q-Agent

    price_thresholds = make_price_threshold(df=train_df)
    water_thresholds = make_water_threshold()

    # Baseline
    run_baseline(train_df, val_df, price_thresholds, water_thresholds)


    # Tabular Q-Agent
    run_tabQ(train_df, val_df, price_thresholds, water_thresholds)




def run_baseline(train_df, df_val, price_thresholds, water_thresholds):
    n_discrete_actions = 3
    # state space => water level (index 0), price (index 1)
    state_space = [10, 4]

    # train
    env = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=df_val, warm_start=False, warm_start_step=2000,
                shaping=False)

    baseline_policy = BaselinePolicy()
    basline_agent = BaselineAgent(env=env, policy=baseline_policy, num_episodes=1, action_prob=1)

    episode_lengths, episode_returns, episode_actions, episode_water = basline_agent.execute_quantiles()
    r_episode_lengths, r_episode_returns, r_episode_actions, r_episode_water =  basline_agent.execute_random()




def run_tabQ(train_df, df_val, price_thresholds, water_thresholds):
   
    n_discrete_actions = 3
    # state space => water level (index 0), price (index 1)
    state_space = [10, 4]

    # train
    env = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=train_df, warm_start=False, warm_start_step=2000,
                shaping=False)

    agent = QAgent(env=env, policy=TabEpsilonGreedyPolicy(), num_episodes=1200, price_threshold=price_thresholds,
               water_threshold=water_thresholds)


    Q, avg_rewards, avg_shaped_rewards, episode_lengths, episode_returns, episode_shaped_returns, viz_data = agent.execute_qlearning(epsilon=0.1, epsilon_end=0.05, adaptive_epsilon = True, adapting_learning_rate = True)

    plt.plot(episode_returns)
    plt.show()

    # validate 
    env = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=df_val)
    agent = QAgent(env=env, policy=TabEpsilonGreedyPolicy(), num_episodes=1000, price_threshold=price_thresholds,
                water_threshold=water_thresholds)


    Q_policy = agent.greedification(Q)
    val_episode_lengths, val_episode_returns, val_viz_data = agent.evaluate_policy(Q_policy)
        
    plt.plot(val_episode_returns)
    plt.show()


def run_deep_q(train_df, df_val, price_thresholds, water_thresholds):
    pass




if __name__ == "__main__":
    main()
