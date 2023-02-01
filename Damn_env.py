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

# Set the seed for reproducibility
seed = 432
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class DamEnv(gym.Env):
    """
    Dam Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_discrete_actions: int, state_space: List[int], price_table: pd.DataFrame,
                 warm_start: bool = False, warm_start_step: int = 200, shaping=False, shaping_type:int = 1) -> None:
        """
        Params of the Dam environment.
        :param n_discrete_actions: The discrete actions.
        :param state_space: The list of the number of categories of each state.
        :param price_table: A pandas df with all prices for energy per hour per month per year.
        :param warm_start: Bool whether the agent
        """
        super(DamEnv, self).__init__()

        self.price_table = price_table
        self.highest_price = self.price_table["Price (Euros)"].max()
        self.water_mean = np.mean(range(0, 100000))
        self.water_std = np.std(range(0, 100000))
        self.price_mean = np.mean(self.price_table["Price (Euros)"])
        self.price_std = np.std(self.price_table["Price (Euros)"])

        # Define action and observation space for the agent
        self.action_space = spaces.Discrete(n_discrete_actions)
        # Example for using image as input:
        self.state_space = state_space
        self.observation_space = spaces.MultiDiscrete(self.state_space)



        # environment space continuous => waterlevel, price
        self.env_observation_space = spaces.Dict({
            "waterlevel": spaces.Box(low=0, high=100000),
            "price": spaces.Box(low=0, high=np.inf)})


        # initial_state
        self.initial_state = (50000, self.price_table["Price (Euros)"][0])
        # current state
        self.current_state = self.initial_state
        # initial time based on index of pd dataframe
        self.initial_time = 0
        # current time
        self.current_time = self.initial_time
        # dam throughput capacity
        self.max_transition_m3h = 5 * 3600
        self.max_transition_kgh = 5 * 1000 * 3600
        # Reservoir size
        self.max_waterlevel = 100000
        # whether warm start and counter
        self.warm_start = warm_start
        self.warm_start_step = warm_start_step
        self.warm_start_counter = 0
        # Reward shaping
        self.shaping = shaping
        # 1 is our shaping, 2 is proposed by vincent
        self.shaping_type = shaping_type



    def step(self, action: int) -> Tuple[Tuple, float, bool, int]:
        """
        Execute one time step within the environment.
        :param action: The action the agent took to make the step.
        :returns the next state, the reward, if the episode is done and the new info.
        """
        # returns next state, reward, done, info
        # next state => water level => based on action; price based on info (time index)
        # reward => profit based on action
        # done => if over; if time time index == len(df) then done
        # info => current time step
        watervolume = self.return_waterlevel(action=action)

        if watervolume == 0:
            action = 1

        # if action 0; sell
        if action == 0:
            next_waterlevel = self.current_state[0] - watervolume
            reward, shaped_reward = self.reward_function(self.current_state, action, watervolume)
            next_state = (next_waterlevel, self.price_table["Price (Euros)"][self.current_time + 1])
            done = self.is_done()
            info = self.current_time + 1

        # if action 1; do nothing
        elif action == 1:
            next_state = (self.current_state[0], self.price_table["Price (Euros)"][self.current_time + 1])
            reward, shaped_reward = 0, 0
            done = self.is_done()
            info = self.current_time + 1

        # If action == 2 (we buy)
        elif action == 2:
            next_waterlevel = self.current_state[0] + watervolume
            reward, shaped_reward = self.reward_function(self.current_state, action, watervolume*1.25)
            next_state = (next_waterlevel, self.price_table["Price (Euros)"][self.current_time + 1])
            done = self.is_done()
            info = self.current_time + 1

        self.update_state(next_state)
        return next_state, reward, shaped_reward, done, info, action

    def update_state(self, next_state: Tuple[float, float]):
        """
        Helper function to update the state variables for the new state.
        :param next_state: The next state the agent will go to
        """
        if self.warm_start:
            self.warm_start_counter += 1

        self.current_state = next_state
        self.current_time += 1

    def return_waterlevel(self, action: int) -> float:
        """
        Returns water volume based on the
        upper or lower waterlevel. Water volume in m3h, action based on action.
        :returns the current waterkevel in m3h.
        """
        water_amount = 0

        # action 0 => sell
        # if waterlevel lower ist als transition; sell left-over; else sell transition
        if action == 0:
            if self.current_state[0] < self.max_transition_m3h:
                water_amount = self.current_state[0]
            else:
                water_amount = self.max_transition_m3h

        # action 2 => buy
        # if free space smaller als transition; buy left-over space; else buy transition space
        elif action == 2:
            if (self.max_waterlevel - self.current_state[0]) < self.max_transition_m3h:
                water_amount = self.max_waterlevel - self.current_state[0]
            else:
                water_amount = self.max_transition_m3h

        return water_amount


    def joule_to_megawatt_hours(self, joule: float) -> float:
        """
        converts joule to megawatt hours.
        :returns the energy in megawatt hours.
        """
        return joule / 3600000000


    def reward_function(self, obs: Tuple[float, float], action: int, waterlevel: float) -> float:
        """
        calculates the reward.
        :param obs: state the reward funtion is based on.
        :param action: Taken action.
        :param waterlevel: water amount that can be sold/bought
        :returns reward in euros.
        """
        reward = 0
        shaped_reward = 0
        #transition_kg = self.max_transition_m3h * 1000  # Calculate max transition in kg

        # if we sell...
        if action == 0:
            U_potential_energy = (waterlevel * 1000) * 9.81 * 30  # Calculate potential energy in Joule
            sellable_energy = U_potential_energy * 0.9  # Actually sellable energy in Joule
            sellable_energy_mgwh = self.joule_to_megawatt_hours(joule=sellable_energy)  # Sellable energy in mgwh
            reward = obs[1] * sellable_energy_mgwh  # price of sellable energy

            # if reward shaping
            if self.shaping == True:
                if self.shaping_type == 1:
                    waterlevel_mgwh = self.joule_to_megawatt_hours(joule=U_potential_energy)
                    shaped_reward = reward - 43 * waterlevel_mgwh

                else:
                    if obs[1] > 48.09:
                        shaped_reward = reward * 2
                    elif obs[1] < 29.59:
                        shaped_reward = reward / 2
                    else:
                        shaped_reward = reward

        # if we buy...
        elif action == 2:
            U_transition = (waterlevel * 1000) * 9.81 * 30
            transition_mgwh = self.joule_to_megawatt_hours(U_transition)
            reward = -(obs[1] * transition_mgwh)

            # if reward shaping
            if self.shaping == True:
                # shaping type one => balancing for waterlevel
                if self.shaping_type == 1:
                    shaped_reward = reward + 43 * transition_mgwh

                # shaping type 2 => incentivize buying for high/low price
                else:
                    if obs[1] > 48.09:
                        shaped_reward = reward * 2
                    elif obs[1] < 29.59:
                        shaped_reward = reward / 2
                    else:
                        shaped_reward = reward

        return (reward, shaped_reward)


    def reset(self, random_reset: bool = False) -> Tuple[float, float]:
        """
         Reset the state of the environment to an initial state. Set Initial state.
        """
        if self.warm_start:
            index = np.random.randint(0, len(self.price_table) - 401)
            self.current_state = (50000, self.price_table["Price (Euros)"][index])
            self.current_time = index
            self.warm_start_counter = 0

        elif random_reset:
            index = np.random.randint(0, len(self.price_table) - 401)
            random_waterlevel = np.random.randint(0, 100000)
            self.current_state = (random_waterlevel, self.price_table["Price (Euros)"][index])
            self.current_time = index

        else:
            self.current_state = self.initial_state
            self.current_time = self.initial_time

        return self.current_state

    def is_done(self) -> bool:
        """
        Checks if the episode is finished.
        :returns bool indicating whether the agent is finishes or not.
        """
        if self.warm_start:
            done = True if self.current_time + 2 == len(self.price_table) or self.warm_start_counter >= self.warm_start_step else False
            if done:
                self.warm_start_counter = 0
        else:
            done = True if self.current_time + 2 == len(self.price_table) else False
        return done

