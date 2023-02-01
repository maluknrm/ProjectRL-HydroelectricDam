import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from typing import Tuple, Dict, List



class BaselineAgent:
    """
    Class for the agent.
    """
    def __init__(self, env, policy, num_episodes, price_threshold,
                 water_threshold, action_prob = 0.7, discount_factor=0.9, alpha=0.5):
        """
        Params for the agent.
        """
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.action_prob = action_prob
        self.price_threshold = price_threshold
        self.water_threshold = water_threshold


    def execute_quantiles(self):
        """
        Implementation of a baseline algorithm. The baseline indicates that we buy if we are
        in the lowest price quantile, we sell if the price is in the highest quantile, if the price is
        in the second-highest price quantile and second-lowest price quantile we sell/buy with a
        percentage of 70%.
        """
        # get stats
        stats = []

        # index
        i = 0

        # reward
        R = 0
        rewards = []

        # actions
        actions = []

        # waterlevel
        waterlevel = []

        for _ in tqdm(range(self.num_episodes)):
            self.env.reset()

            while True:
                disc_state = self.discretize_state(self.env.current_state)
                action = self.policy.baseline_policy(state=disc_state, action_prob=self.action_prob)
                new_state, reward, _, done, info, action = self.env.step(action)

                R += reward
                rewards.append(R)

                i += 1
                actions.append(action)
                waterlevel.append(self.env.current_state[0])

                if done:
                    break

            # append stats
            stats.append((i, rewards, actions, waterlevel))

        episode_lengths, episode_returns, episode_actions, episode_water = zip(*stats)
        return episode_lengths, episode_returns, episode_actions, episode_water


    def execute_random(self):
        """
        Implementation of a baseline algorithm. The baseline indicates that we buy if we are
        in the lowest price quantile, we sell if the price is in the highest quantile, if the price is
        in the second-highest price quantile and second-lowest price quantile we sell/buy with a
        percentage of 70%.
        """

        # get stas
        stats = []

        # index
        i = 0

        # reward
        R = 0
        rewards = []

        # actions
        actions = []

        # waterlevel
        waterlevel = []

        for _ in tqdm(range(self.num_episodes)):
            self.env.reset()

            while True:
                disc_state = self.discretize_state(self.env.current_state)
                action = self.policy.random_policy(state=disc_state)
                new_state, reward, _, done, info, action = self.env.step(action)

                R += reward
                rewards.append(R)

                i += 1
                actions.append(action)
                waterlevel.append(self.env.current_state[0])

                if done:
                    break

            # append stats
            stats.append((i, rewards, actions, waterlevel))

        episode_lengths, episode_returns, episode_actions, episode_water = zip(*stats)
        return episode_lengths, episode_returns, episode_actions, episode_water

    def discretize_state(self, state: Tuple[int, float]) -> Tuple[int, int]:
        """
        Discretizes the continuous state.
        :param state: Continuous state
        :returns the state in a discretized form
        """
        disc_water = np.digitize(state[0], self.water_threshold, right=True)
        disc_prize = np.digitize(state[1], self.price_threshold, right=True)

        return disc_water, disc_prize



class BaselinePolicy():
    """
    Class for the policies.
    """

    def baseline_policy(self, state, action_prob):
        """
        Baseline policy based on state and action.
        The baseline indicates that we buy if we are
        in the lowest price quantile, we sell if the price is in the highest quantile, if the price is
        in the second-highest price quantile and second-lowest price quantile we sell/buy with a
        percentage of 70%.
        :param state: discretized state that determines the action
        : returns action
        """
        action = 1
        state_price = state[1]
        random_prob = np.random.uniform(0, 1)

        # if price cheap we buy => 2
        if state_price == 0:
            action = 2

        # if a bit cheap we buy with probability action prob
        elif state_price == 1:
            if random_prob > action_prob:
                action = 2

        # if a bit expensive we sell with probability action prob => 0
        elif state_price == 2:
            if random_prob > action_prob:
                action = 0

        # if expensive we sell => 0
        elif state_price == 3:
            action = 0

        return action

    def random_policy(self, state):
        """
        Completely random baseline.

        :param state: discretized state that determines the action
        :return: action
        """
        return random.sample([0, 1, 2], k=1)[0]