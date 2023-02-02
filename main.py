import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from data_transformation import DataTransformation
from baselines import *
from thresholds import *
from Dam_env import *
from TabQAgent import *
from Tab_EpsilonGreedy import *
from TestEnv import *

# TODO Make Argparser
# TODO insert Vincents test env

# TODO make requirements file ?

# TODO implement runs
# TODO look at evaluate
# TODO look at threshold
# TODO write report



# parser = argparse.ArgumentParser(description='Arguments for the RL Projects')
#
# parser.add_argument('-P', "--path",
#                     dest="path", type=str,
#                     help='Testing file path.')
#
#
# args = parser.parse_args()


def main():

    # Data transformation
    data_transformer = DataTransformation()

    train_path = "train.xlsx"
    #val_path = args.path
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
    runs = 2

    run_tabQ(runs, train_df, train_path, val_path, val_df, price_thresholds, water_thresholds)




def run_baseline(train_df, val_df, price_thresholds, water_thresholds):
    print(40 * "-")
    print("Run baseline")

    n_discrete_actions = 3
    # state space => water level (index 0), price (index 1)
    state_space = [10, 4]

    # validate
    env = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=val_df, warm_start=False, warm_start_step=2000,
                shaping=False)

    baseline_policy = BaselinePolicy()
    basline_agent = BaselineAgent(env=env, policy=baseline_policy,
                                  num_episodes=1, price_threshold=price_thresholds, water_threshold=water_thresholds)

    episode_lengths, episode_returns, episode_actions, episode_water = basline_agent.execute_quantiles()
    r_episode_lengths, r_episode_returns, r_episode_actions, r_episode_water = basline_agent.execute_random()

    print("Results:")
    print(f"The return after applying the qunatiles baseline to the testing dataframe was: {episode_returns[-1][-1]}")




def run_tabQ(runs, train_df, train_path, val_path, val_df, price_thresholds, water_thresholds):
    print(40 * "-")
    print("Run Tabular Q-Learning")

    n_discrete_actions = 3
    state_dim = 2
    state_space = [10, 4]
    #state_space = [10, 4, 3]
    num_episodes = 1000
    reward_shaping_bool = False
    reward_shaping_type = 2


    dict_run = {
        "Run": [run_list_item for run_list in [[run] * 2 * num_episodes for run in range(0, runs)] for run_list_item in
                run_list],
        "Episode": list(range(0, num_episodes)) * 2 * runs,
        "Return": [],
        "Stage": list(["Training"] * num_episodes + ["Validation"] * num_episodes) * runs}


    for _ in range(runs):

        # environemnt
        # train
        env_train_vincent = HydroElectric_Test(path_to_test_data=train_path)
        env = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=train_df, warm_start=False, warm_start_step=2000,
                    shaping=reward_shaping_bool, shaping_type=reward_shaping_type)

        # validate with our env
        env_val = DamEnv(n_discrete_actions=n_discrete_actions, state_space=state_space, price_table=val_df)

        # validate with Vincents env
        env_val_vincent = HydroElectric_Test(path_to_test_data=val_path)

        agent = QAgent(env=env, policy=TabEpsilonGreedyPolicy(), num_episodes=num_episodes, price_threshold=price_thresholds,
                   water_threshold=water_thresholds, state_dim=state_dim, alpha=0.5)


        Q, avg_rewards, avg_shaped_rewards, episode_lengths, episode_returns, episode_shaped_returns, viz_data, val_returns = agent.execute_qlearning(validation_env=env_val, adaptive_epsilon = True, adapting_learning_rate = False)

        concat_return = list(episode_returns) + [element[0] for element in val_returns]
        dict_run["Return"] += concat_return


        # plot for us
        # plt.plot(episode_returns, label='Train Return')
        # plt.plot(val_returns, label='Validation Return')
        # plt.xlabel("Number of Episodes")
        # plt.ylabel("Return")
        # plt.legend()
        # plt.title("Training return with reward shaping")
        # plt.show()

        Q_policy = agent.greedification(Q)
        val_episode_lengths, val_episode_returns, val_viz_data = agent.evaluate_policy_vincent(env_val_vincent, Q_policy)

        #plt.plot(val_episode_returns)
        #plt.show()
        print("Results:")
        print(f"The return of the test set after applying the Tabular Q-Learning after {num_episodes} episodes was: {val_episode_returns[-1]}")

        val_episode_lengths, val_episode_returns, val_viz_data = agent.evaluate_policy(env_val, Q_policy)

        print("Results:")
        print(f"The return of the test set after applying the Tabular Q-Learning after {num_episodes} episodes was: {val_episode_returns[-1]}")
        visualise_q(agent, Q, state_dim, reward_shaping_bool, reward_shaping_type)

    #print(dict_run)
    df = pd.DataFrame(data=dict_run)
    df.to_csv(f"runs/Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}")

def visualise_q(agent, Q, state_dim, reward_shaping_bool, reward_shaping_type):
    Q_policy = agent.greedification(Q)

    if agent.state_dim == 2:
        Q_max_vals = Q.max(axis=2)
        np.savetxt(f"results/Q_vals_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q_max_vals, delimiter=",")
        np.savetxt(f"results/Q_policy_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q_policy, delimiter=",")

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

        np.savetxt(f"results/Q_val_table_Q0_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q0_max, delimiter=",")
        np.savetxt(f"results/Q_val_table_Q1_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q1_max, delimiter=",")
        np.savetxt(f"results/Q_val_table_Q2_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q2_max, delimiter=",")

        Q0_eval = np.squeeze(Q[:, :, :1])
        Q1_eval = np.squeeze(Q[:, :, 1:2])
        Q2_eval = np.squeeze(Q[:, :, 2:])

        Q0_pol = Q0_eval.argmax(axis=2)
        Q1_pol = Q1_eval.argmax(axis=2)
        Q2_pol = Q2_eval.argmax(axis=2)

        np.savetxt(f"results/Q_val_policy_Q0_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q0_pol, delimiter=",")
        np.savetxt(f"results/Q_val_policy_Q1_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q1_pol, delimiter=",")
        np.savetxt(f"results/Q_val_policy_Q2_Statedim_{state_dim}_rewardshaping_{reward_shaping_bool}_type_{reward_shaping_type}", Q2_pol, delimiter=",")




def run_deep_q(train_df, df_val, price_thresholds, water_thresholds):
    pass




if __name__ == "__main__":
    main()
