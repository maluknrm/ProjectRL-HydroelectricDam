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
# TODO do baseline
# TODO implement runs
# TODO




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
    #run_baseline(train_df, val_df, price_thresholds, water_thresholds)


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

    #state_space = [10, 4]
    state_space = [10, 4, 3]

    # train
    env = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=train_df, warm_start=False, warm_start_step=2000,
                shaping=True, shaping_type=2)

    # validate
    env_val = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=df_val)

    agent = QAgent(env=env, policy=TabEpsilonGreedyPolicy(), num_episodes=1200, price_threshold=price_thresholds,
               water_threshold=water_thresholds, state_dim=3)


    Q, avg_rewards, avg_shaped_rewards, episode_lengths, episode_returns, episode_shaped_returns, viz_data, val_returns = agent.execute_qlearning(validation_env=env_val, adaptive_epsilon = True, adapting_learning_rate = True)

    plt.plot(episode_returns)
    plt.plot(val_returns)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Return")
    plt.title("Training return with reward shaping")
    plt.show()

    # plt.plot(val_returns)
    # plt.xlabel("Number of Episodes")
    # plt.ylabel("Return")
    # plt.title("Validation return with reward shaping")
    # plt.show()

    # validate
    agent = QAgent(env=env, policy=TabEpsilonGreedyPolicy(), num_episodes=1000, price_threshold=price_thresholds,
                water_threshold=water_thresholds, state_dim=3)

    if agent.state_dim == 2:
        Q_max_vals = Q.max(axis=2)
        np.savetxt(f"results/Q_val_table_space_{agent.state_dim}.csv", Q_max_vals, delimiter=",")
        Q_policy = agent.greedification(Q)
        np.savetxt(f"results/Q_val_policy_space_{agent.state_dim}.csv", Q_policy, delimiter=",")
        val_episode_lengths, val_episode_returns, val_viz_data = agent.evaluate_policy(Q_policy)
        print(val_episode_returns)

    # Visualisation
    elif agent.state_dim == 3:
        #make 3 Q tables based on trend
        Q0 = np.squeeze(Q[:, :, :1, ])
        Q1 = np.squeeze(Q[:, :, 1:2, ])
        Q2 = np.squeeze(Q[:, :, 2:, ])

        # take max Q-value for each
        Q0_max = Q0.max(axis=2)
        Q1_max = Q1.max(axis=2)
        Q2_max = Q2.max(axis=2)

        np.savetxt(f"results/Q_val_table_Q0_space_{agent.state_dim}.csv", Q0_max, delimiter=",")
        np.savetxt(f"results/Q_val_table_Q1_space_{agent.state_dim}.csv", Q1_max, delimiter=",")
        np.savetxt(f"results/Q_val_table_Q2_space_{agent.state_dim}.csv", Q2_max, delimiter=",")

        Q_policy = agent.greedification(Q)
        Q0_eval = np.squeeze(Q[:, :, :1])
        Q1_eval = np.squeeze(Q[:, :, 1:2])
        Q2_eval = np.squeeze(Q[:, :, 2:])

        Q0_pol = Q0_eval.argmax(axis=2)
        Q1_pol = Q1_eval.argmax(axis=2)
        Q2_pol = Q2_eval.argmax(axis=2)

        np.savetxt(f"results/Q_val_policy_Q0_val_space_{agent.state_dim}.csv", Q0_pol, delimiter=",")
        np.savetxt(f"results/Q_val_policy_Q1_val_space_{agent.state_dim}.csv", Q1_pol, delimiter=",")
        np.savetxt(f"results/Q_val_policy_Q2_val_space_{agent.state_dim}.csv", Q2_pol, delimiter=",")

        val_episode_lengths, val_episode_returns, val_viz_data = agent.evaluate_policy(env_val, Q_policy)
        print(val_episode_returns)



def run_deep_q(train_df, df_val, price_thresholds, water_thresholds):
    pass




if __name__ == "__main__":
    main()
