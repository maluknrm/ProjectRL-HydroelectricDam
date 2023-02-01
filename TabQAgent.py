import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Tuple
from numpy.typing import NDArray


class QAgent():
    """
    Class for the Q-learning agent with two states.
    """

    def __init__(self, env, policy, num_episodes, price_threshold, water_threshold, state_dim: int = 2, discount_factor=0.99, alpha=0.1):
        """
        Params for the agent.
        :param env: The environment.
        :param policy: The policy to sample the action.
        :param num_episodes: The number of episodes the Q-learning should get evaluated.
        :param price_threshold: The thresholds to discretize the price state.
        :param water_threshold: The thresholds to discretize the water state.
        :param discount_factor: The used discount factor.
        :param alpha: The used learning rate
        """
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.state_dim = state_dim

        self.price_threshold = price_threshold
        self.water_threshold = water_threshold
        self.discount_factor = discount_factor

        self.action_space = self.env.action_space.n

        shape = self.env.state_space + [self.action_space]
        self.Q = np.zeros(shape)


    def discretize_state(self,  state, old_state):
        """
        Main function for getting the right function.
        """
        if self.state_dim == 2:
            state = self._discretize_state_2(state, old_state)

        elif self.state_dim == 3:
            state = self._discretize_state_3(state, old_state)
        
        elif self.state_dim == 4:
            state = self._discretize_state_4(state, old_state)
        
        return state


    def _discretize_state_2(self, state: Tuple[int, float], old_state) -> Tuple[int, int]:
        """
        Discretizes the continuous state for two state dimensions.
        :param state: Continuous state.
        :returns the state in a discretized form.
        """
        disc_water = np.digitize(state[0], self.water_threshold, right=True)
        disc_prize = np.digitize(state[1], self.price_threshold, right=True)

        return disc_water, disc_prize

    
    
    def _discretize_state_3(self, state: Tuple[int, float], old_state) -> Tuple[int, int]:
        """
        Discretizes the continuous state.
        :param state: Continuous state.
        :returns the state in a discretized form.
        """
        disc_water = np.digitize(state[0], self.water_threshold, right=True)
        disc_prize = np.digitize(state[1], self.price_threshold, right=True)

        if old_state == (0, 0):
            disc_trend = 0
        
        else:
            trend = state[1] - old_state[1]

            if trend > 0:
                disc_trend = 2
            elif trend < 0:
                disc_trend = 0
            else:
                disc_trend = 1

        return disc_water, disc_prize, disc_trend


    def _discretize_state_4(self, state: Tuple[int, float], old_state) -> Tuple[int, int]:
        """
        Discretizes the continuous state.
        :param state: Continuous state.
        :returns the state in a discretized form.
        """
        disc_water = np.digitize(state[0], self.water_threshold, right=True)
        disc_prize = np.digitize(state[1], self.price_threshold, right=True)

        return disc_water, disc_prize, state[2], state[3]




    def execute_qlearning(self,
                          validation_env,
                          epsilon: float = 0.3, epsilon_end: float = 0.05,
                          adaptive_epsilon: bool = False,
                          adapting_learning_rate: bool = False) -> Tuple[NDArray, List, List, List]:
        """
        Executes the Q-Learning algorithm.
        :param epsilon: The epsilon for the epsilon-greedy policy.
        :param epsilon_end: The smallest epsilon for the adaptive epsilon-greedy policy.
        :param adaptive_epsilon: Bool whether adaptive epsilon is used or not.
        :param adapting_learning_rate: Bool whether adaptive epsilon is used or not.
        :returns the Q-table, the avreage rewards, the episode lengths and the episode returns.
        """

        # keep track of stats
        stats = []
        rewards = []
        avg_rewards = []
        avg_shaped_rewards = []
        shaped_rewards = []
        val_returns = []

        # If adaptive learning rate, it starts with a value of 1 and decays it over time
        if adapting_learning_rate:
            self.alpha = 1

        for i in tqdm(range(self.num_episodes)):

            # Initialize the state
            state = self.env.reset()
            # first old state (dummy)
            old_state = (0, 0)

            # Start recording viz_data once we reach last episode
            viz_prices = []
            viz_actions = []
            viz_waterlevels = []
            viz_rewards = [0]
            viz_shaped_rewards = [0]
            record_viz = False

            if i == self.num_episodes - 1:
                record_viz = True
                waterlevel = state[0]
                price = state[1]
                viz_waterlevels.append(waterlevel)
                viz_prices.append(price)

            # Discretize the state space regarding the dimension
            undisc_state = state
            state = self.discretize_state(state, old_state)

            # Keep track of return and rewards
            R = 0
            R_shaped = 0

            # keep track of the episodes
            i_episode = 0

            # If adaptive epsilon rate
            if adaptive_epsilon:
                epsilon = self.policy.get_epsilon(iteration=i, epsilon_end=epsilon_end)

            # loop until the episode is terminated
            while True:

                action = self.policy.sample_action(obs=state, epsilon=epsilon, Q=self.Q, state_dim=self.state_dim)

                # next transition
                new_state, reward, shaped_reward, done, _, taken_action = self.env.step(action)
                waterlevel = new_state[0]
                price = new_state[1]
                undisc_new_state = new_state
                new_state = self.discretize_state(new_state, undisc_state)

                # update the return
                R += reward
                R_shaped += shaped_reward
                i_episode += 1

                # Record viz data of last episode
                if record_viz == True:
                    viz_prices.append(price)
                    viz_actions.append(taken_action)
                    viz_waterlevels.append(waterlevel)
                    viz_rewards.append(reward)
                    viz_shaped_rewards.append(shaped_reward)

                    # waterlevel = new_state[0]
                    # price = new_state[1]

                # Get new action
                new_action = self.policy.sample_action(obs=new_state, epsilon=epsilon, Q=self.Q, state_dim=self.state_dim)

                # update Q-value
                self.update_q(state, new_state, action, new_action, reward, shaped_reward)


                # check if the episode is over
                if done:
                    policy = self.greedification(self.Q)
                    val_episode_lengths, val_episode_returns, val_viz_data = self.evaluate_policy(validation_env, policy)
                    val_returns.append(val_episode_returns)
                    break

                undisc_state = undisc_new_state
                state = new_state

            if adapting_learning_rate:
                self.alpha = self.alpha / np.sqrt(i + 1)

            rewards.append(R)
            shaped_rewards.append(R_shaped)
            stats.append((i_episode, R, R_shaped))

            # Prep visualisation data
            viz_actions.append(None)
            viz_data = {"Price": viz_prices, "Waterlevel": viz_waterlevels, "Reward": viz_rewards,
                        "Shaped Reward": viz_shaped_rewards, "Action": viz_actions}

            # Calculate the average score over 100 episodes
            if i % 100 == 0:
                avg_rewards.append(np.mean(rewards))
                avg_shaped_rewards.append(np.mean(shaped_rewards))

                # Initialize a new reward list, as otherwise the average values would reflect all rewards!
                rewards = []
                shaped_rewards = []

        episode_lengths, episode_returns, episode_shaped_returns = zip(*stats)
        return self.Q, avg_rewards, avg_shaped_rewards, episode_lengths, episode_returns, episode_shaped_returns, viz_data, val_returns


    
    def update_q(self, state, new_state, action, new_action, reward, shaped_reward):
        """
        Function for updating the Q value.
        """
        if self.state_dim == 2:
            self.Q[state[0], state[1], action] = self._update_q_2(Q=self.Q, state=state, 
                                                    new_state=new_state, action=action, 
                                                    new_action=new_action, reward=reward, 
                                                    shaped_reward=shaped_reward)

        elif self.state_dim == 3:
            self.Q[state[0], state[1], state[2], action] = self._update_q_3(Q=self.Q, state=state, 
                                                    new_state=new_state, action=action, 
                                                    new_action=new_action, reward=reward, 
                                                    shaped_reward=shaped_reward)
        
        elif self.state_dim == 4:
            self.Q[state[0], state[1], state[2], state[3], action] = self._update_q_4(Q=self.Q, state=state, 
                                                    new_state=new_state, action=action, 
                                                    new_action=new_action, reward=reward, 
                                                    shaped_reward=shaped_reward)

        

     
  
    def _update_q_2(self, Q, state, new_state, action, new_action, reward, shaped_reward):
        """
        Function for updating the Q value.
        """
        # Get Q-value
        old_q = Q[state[0], state[1], action]
        new_q = Q[new_state[0], new_state[1], new_action]

        if self.env.shaping == False:
            update_value = old_q + self.alpha * (reward + self.discount_factor*new_q - old_q)
        else:
            update_value = old_q + self.alpha * (shaped_reward + self.discount_factor*new_q - old_q)

        return update_value

    def _update_q_3(self, Q, state, new_state, action, new_action, reward, shaped_reward):
        """
        Function for updating the Q value.
        """
        # Get Q-value
        old_q = Q[state[0], state[1], state[2], action]
        new_q = Q[new_state[0], new_state[1], state[2], new_action]

        if self.env.shaping == False:
            update_value = old_q + self.alpha * (reward + self.discount_factor*new_q - old_q)
        else:
            update_value = old_q + self.alpha * (shaped_reward + self.discount_factor*new_q - old_q)

        return update_value


    def _update_q_4(self, Q, state, new_state, action, new_action, reward, shaped_reward):
        """
        Function for updating the Q value.
        """
        # Get Q-value
        old_q = Q[state[0], state[1], state[2], state[3], action]
        new_q = Q[new_state[0], new_state[1], state[2], state[3], new_action]

        if self.env.shaping == False:
            update_value = old_q + self.alpha * (reward + self.discount_factor*new_q - old_q)
        else:
            update_value = old_q + self.alpha * (shaped_reward + self.discount_factor*new_q - old_q)

        return update_value


    def get_greedy_action(self, policy, state):
        if self.state_dim == 2:
            action = policy[state[0], state[1]]
        
        elif self.state_dim == 3:
            action = policy[state[0], state[1], state[2]]

        elif self.state_dim == 4:
            action = policy[state[0], state[1], state[2], state[3]]
        
        return action


    def evaluate_policy(self, validation_env, policy: NDArray) -> Tuple[List, List]:
        """
        Evaluates the policy on an evaluation dataframe.
        :param policy: A learned policy.
        :returns The validation episode lengths and returns.
        """

        # keep track of stats
        stats = []

        # Initialize the state
        state = validation_env.reset()
        old_state = (0, 0)

        # save information for visulization
        viz_actions = []
        viz_waterlevels = []
        viz_rewards = [0]
        viz_prices = []
        waterlevel = state[0]
        price = state[1]
        viz_waterlevels.append(waterlevel)
        viz_prices.append(price)

        # Discretize the state space
        undisc_state = state
        state = self.discretize_state(state, old_state)

        # Keep track of return and rewards
        R = 0

        # keep track of the episodes
        i_episode = 0

        # loop until the episode is terminated
        while True:
            action = self.get_greedy_action(policy, state)

            # next transition
            new_state, reward, shaped_rewards, done, _, taken_action = validation_env.step(action)
            waterlevel = new_state[0]
            price = new_state[1]
            undisc_new_state = new_state
            new_state = self.discretize_state(new_state, undisc_state)

            # update the return
            R += reward
            i_episode += 1

            # save visualisation data
            viz_prices.append(price)
            viz_actions.append(taken_action)
            viz_waterlevels.append(waterlevel)
            viz_rewards.append(reward)
            waterlevel = new_state[0]
            price = new_state[1]

            
            # check if the episode is over
            if done:
                break

            undisc_state = undisc_new_state
            state = new_state

        stats.append((i_episode, R))

        viz_actions.append(None)
        viz_data = {"Price": viz_prices, "Waterlevel": viz_waterlevels, "Reward": viz_rewards, "Action": viz_actions}

        episode_lengths, episode_returns = zip(*stats)
        return episode_lengths, episode_returns, viz_data

    
    def greedification(self, Q_table: NDArray) -> NDArray:
        """
        Changes the learned Q-table into a policy matrix using greedification.
        :param Q_table: The learned Q-table.
        :returns The policy.
        """
        axis = len(np.shape(Q_table)) - 1
        policy = Q_table.argmax(axis=axis)
        return policy
    

    def evaluate_policy_vincent(self, TestEnv, policy: NDArray) -> Tuple[List, List]:
        """
        Evaluates the policy on an evaluation dataframe.
        :param policy: A learned policy.
        :returns The validation episode lengths and returns.
        """

        # keep track of stats
        stats = []

        # Initialize the state
        state = TestEnv.observation()[:2]
        old_state = (0, 0)

        # save information for visulization
        viz_actions = []
        viz_waterlevels = []
        viz_rewards = [0]
        viz_prices = []
        waterlevel = state[0]
        price = state[1]
        viz_waterlevels.append(waterlevel)
        viz_prices.append(price)

        # Discretize the state space
        state = self.discretize_state(state, old_state)

        # Keep track of return and rewards
        R = 0

        # keep track of the episodes
        i_episode = 0

        # loop until the episode is terminated
        while True:

            action = self.get_greedy_action(policy, state)

            # modify to Vincents actions
            vincent_action = 0
            if action == 0:
                vincent_action = -1
            elif action == 1:
                vincent_action = 0
            elif action == 2:
                vincent_action = 1

            # next transition
            new_state, reward, terminated, truncated, _ = TestEnv.step(vincent_action)
            new_state = new_state[:2]

            waterlevel = new_state[0]
            price = new_state[1]
            new_state = self.discretize_state(new_state, state)

            # update the return
            R += reward
            i_episode += 1

            # save visualisation data
            viz_prices.append(price)
            viz_waterlevels.append(waterlevel)
            viz_rewards.append(reward)
            waterlevel = new_state[0]
            price = new_state[1]

            # check if the episode is over
            if terminated or truncated:
                break

            state = new_state

        stats.append((i_episode, R))

        viz_actions.append(None)
        viz_data = {"Price": viz_prices, "Waterlevel": viz_waterlevels, "Reward": viz_rewards, "Action": viz_actions}

        episode_lengths, episode_returns = zip(*stats)
        return episode_lengths, episode_returns, viz_data
