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
from TestEnv import *

# Set the seed for reproducibility
seed = 432
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)





class QNetwork(nn.Module):

    def __init__(self, learning_rate: float, input_features: int, num_hidden=128):
        """
        Params:
        :param env: environment that the agent needs to play
        :param learning_rate: learning rate used in the update
        """
        nn.Module.__init__(self)
        input_features = input_features
        action_space = env.action_space.n
        self.l1 = nn.Linear(in_features = input_features, out_features = num_hidden)
        self.l2 = nn.Linear(in_features = num_hidden, out_features = 64)
        self.l3 = nn.Linear(in_features = 64, out_features = action_space)

        # Initialise ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Makes a forward pass.
        :param x: observation
        """
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """

    def sample_action(self, Q: QNetwork, obs: torch.FloatTensor, epsilon: float) -> int:
        """
        This method takes a state as input and returns an action sampled from this policy.
        :param obs: current state
        returns: An action (int).
        """
        obs = torch.FloatTensor(obs)
        # obs[:1] /= 100000
        # obs[1:2] /= 2500

        # normalisation with mean and std
        obs[:1] = (obs[:1] - 49999.5) / 28867.513458037913
        obs[1:2] = (obs[1:2] - 50.602546760948904) / 40.0892872890141


        with torch.no_grad():
            random_prob = np.random.uniform(0, 1)
            actions = Q(obs)

            if random_prob < epsilon:
                action = np.random.randint(0, len(actions))
            else:
                action = torch.argmax(actions).item()

        return action

    def get_epsilon(self, it: int, epsilon_start: float, epsilon_end: float) -> float:
        """
        Returns epsilon.
        :param it: number of iteration
        :param epsilon_end: lowest possible epsilon
        :param epsilon_start: highest possible epsilon
        :returns the new epsilon
        """
        return epsilon_start - (it*0.00095) if it <= 1000 else epsilon_end

class ExperienceReplay:

    def __init__(self, env: DamEnv, buffer_capacity: int, min_replay_size: int = 3000):
        """
        Parameter for the ExperienceReplay class:
        :param env: environment that the agent needs to play
        :param buffer_size: max number of transitions that the experience replay buffer can store
        :param min_replay_size: min number of (random) transitions that the replay buffer needs to have when initialized
        """
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_capacity)
        self.reward_buffer = deque([0], maxlen = 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs = self.env.reset(random_reset=True)
        for _ in range(self.min_replay_size):
            action = env.action_space.sample()
            new_obs, reward, shaped_reward, done, _, taken_action = env.step(action)

            # try transition with taken action
            transition = (obs, action, reward, shaped_reward, done, new_obs)
            #transition = (obs, taken_action, reward, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs = env.reset(random_reset=True)

            #elif _ % 4 == 0:
                #obs = env.reset(random_reset=True)

    def add_data(self, data: Tuple):
        """
        Appends data to the replay buffer.
        :param data: relevant data of a transition, i.e. action, new_obs, reward, done
        """
        self.replay_buffer.append(data)

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Takes a random sample of the replay buffer.
        :param batch_size: number of transitions that will be sampled
        :returns a sample list with all transitions
        """
        return random.sample(self.replay_buffer, batch_size)

    def add_reward(self, reward: float):
        """
        Add reward to reward buffer.
        :param reward: reward that the agent earned during an episode of a game
        """
        self.reward_buffer.append(reward)


class DDQNAgent:

    def __init__(self, env, Qnet, device, epsilon_greedy,
                 epsilon_start, epsilon_end, discount_rate, alpha, Buffer, buffer_capacity):
        """
        Parameters of the DDQN Agent class.
        :param env: the environment that the agent needs to play
        :param Qnet: the used deep Q-learning network
        :param device: set up to run CUDA operations
        :param epsilon_greedy: class with the epsilon_greedy policy
        :param epsilon_start: starting value for the epsilon value
        :param epsilon_end: ending value for the epsilon value
        :param discount_rate: discount rate for future rewards
        :param alpha: learning rate
        :param buffer: experience replay buffer
        :param buffer_size: max number of transitions that the experience replay buffer can store
        """
        self.env = env
        self.Qnet = Qnet
        self.device = device
        self.discount_rate = discount_rate
        self.alpha = alpha
        # epsilon greedy
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        # initialise replay memory
        self.buffer_capacity = buffer_capacity
        self.replay_memory = Buffer(self.env, self.buffer_capacity)

        # initialise online Q-network
        self.online_network = self.Qnet(self.alpha, len(self.env.observation_space)).to(self.device)

        # initialise target Q-network
        self.target_network = self.Qnet(self.alpha, len(self.env.observation_space)).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def choose_action(self, step: int, observation: torch.FloatTensor, greedy: bool = False) -> int:
        """
        Choose an action according to a policy.
        :param step: the step number.
        :param observation: the observation input
        :param greedy: boolean whether greedy policy is used
        :returns: an action integer.
        """
        if greedy:
            epsilon = 0
            action = self.epsilon_greedy.sample_action(Q=self.online_network, obs=observation, epsilon=epsilon)

        else:
            #epsilon = self.epsilon_greedy.get_epsilon(it=step, epsilon_start=self.epsilon_start, epsilon_end=self.epsilon_end)
            epsilon = np.interp(step, [0, 100000], [self.epsilon_start, self.epsilon_end])
            #epsilon = 0.1
            action = self.epsilon_greedy.sample_action(Q=self.online_network, obs=observation, epsilon=epsilon)

        return action, epsilon


    def return_max_q_value(self, observation: torch.FloatTensor) -> float:
        """
        Get the highest Q-value from an observation.
        :param observation: input value of the state the agent is in
        :returns:  maximum q value
        """
        # for plotting 3d graph
        obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
        q_values = self.online_network(obs_t.unsqueeze(0))

        return torch.max(q_values).item()


    def return_policy_Qvalues(self):

        q_values = {"Waterlevel": [], "Price": [], "Max Q-Value": []}
        self.env.warm_start = False
        state = self.env.reset()
        step = 0

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float)
            state_tensor[:1] = (state_tensor[:1] - self.env.water_mean) / self.env.water_std
            state_tensor[1:2] = (state_tensor[1:2] - self.env.price_mean) / self.env.price_std

            max_qval = self.return_max_q_value(state)
            q_values["Waterlevel"].append(state[0])
            q_values["Price"].append(state[1])
            q_values["Max Q-Value"].append(max_qval)

            action, epsilon = self.choose_action(step=step, observation=state, greedy=True)
            next_state, reward, shaped_reward, done, _, taken_actions = self.env.step(action)
            state = next_state

            if done:
                break

        return q_values


    def train(self, batch_size: int):
        """
        Function to learn optimal Q-values.
        :param batch_size: number of transitions that will be sampled.
        :param global_step:
        """
        transitions = self.replay_memory.sample(batch_size)
        state, action, reward, shaped_reward, done, next_state = zip(*transitions)

        # convert to PyTorch and define types
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)[:, None]
        shaped_reward = torch.tensor(shaped_reward, dtype=torch.float)[:, None]
        done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean

        # normalizing
        # state[:, :1] /= 100000
        # state[:, 1:2] /= self.env.highest_price
        # next_state[:, :1] /= 100000
        # next_state[:, 1:2] /= self.env.highest_price
        # reward /= self.env.highest_price
        # shaped_reward /= self.env.highest_price #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # normalisation with mean and std
        state[:, :1] = (state[:, :1] - self.env.water_mean) / self.env.water_std
        state[:, 1:2] = (state[:, 1:2] - self.env.price_mean) / self.env.price_std
        next_state[:, :1] = (next_state[:, :1] - self.env.water_mean) / self.env.water_std
        next_state[:, 1:2] = (next_state[:, 1:2] - self.env.price_mean) / self.env.price_std
        reward = (reward - self.env.price_mean) / self.env.price_std
        shaped_reward = (shaped_reward - self.env.price_mean) / self.env.price_std

        # Compute targets
        target_q_values = self.target_network(next_state)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        # check if reward shaping is turned on for the updates
        if self.env.shaping:
            targets = shaped_reward + self.discount_rate * (1 - done) * max_target_q_values
        else:
            targets = reward + self.discount_rate * (1-done) * max_target_q_values

        # Compute loss
        q_values = self.online_network(state)
        action_q_values = torch.gather(input=q_values, dim=1, index=action)

        # Loss
        loss = F.smooth_l1_loss(action_q_values, targets)
        # loss = F.mse_loss(action_q_values, targets.detach())

        # backpropagation of loss to Neural Network
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

    def update_target_network(self):
        """
        Updates the target network with the parameters of the online network.
        """
        self.target_network.load_state_dict(self.online_network.state_dict())


    def run_episodes(self, num_episodes: int, validation_env: DamEnv, batch_size: int=1, target_update_freq: int=200) -> List:
        """
        Runs the Double deep Q-learning algorithm.
        :param num_episodes: The number of episodes the agent should play.
        :param validation_env: The environment to evaluate the policy on.
        :param batch_size: The batch size to sample from the memory replay
        :param target_update_freq: Frequency how often to update the target network.
        """
        # for adaptive epsilon
        global_steps = 0

        # track learning curves
        learning_curves = defaultdict(list)

        # running the episodes
        for i in tqdm(range(num_episodes)):
            
            # Reset state
            state = self.env.reset()

            # Keeping track of episode data
            train_viz_data = defaultdict(dict)
            train_viz_data[f"E{i}"] = {
                "Waterlevel": [state[0]],
                "Price": [state[1]], 
                "Action": [],
                "Taken Action": [],
                "Reward": [0],
                "Shaped Reward": [0]}

            # Episode loop
            while True:
                action, epsilon = self.choose_action(step=global_steps, observation=state, greedy=False)
                next_state, reward, shaped_reward, done, _, taken_action = self.env.step(action)

                self.replay_memory.add_data((state, action, reward, shaped_reward, done, next_state))

                global_steps += 1
                state = next_state

                # Save episode stats
                train_viz_data[f"E{i}"]["Waterlevel"].append(state[0])
                train_viz_data[f"E{i}"]["Price"].append(state[1])
                train_viz_data[f"E{i}"]["Action"].append(action)
                train_viz_data[f"E{i}"]["Taken Action"].append(taken_action)
                train_viz_data[f"E{i}"]["Reward"].append(reward)
                train_viz_data[f"E{i}"]["Shaped Reward"].append(shaped_reward)

                # Add return to replay memory
                if done:
                    self.replay_memory.add_reward(sum(train_viz_data[f"E{i}"]["Reward"]))

                    # evaluate validation
                    val_viz_data = self.evaluate(validation_env)

                    # print statement to keep track of development
                    if i % 10 == 0:
                        print(30*'--')
                        print('Epsilon:', round(epsilon, 2))
                        print('Avg Rew:', round(np.mean(self.replay_memory.reward_buffer), 1))
                        print("Training Return of last episode:", round(sum(train_viz_data[f"E{i}"]["Reward"]), 1))
                        print("Validation Return:", round(sum(val_viz_data["Reward"]), 1))
                        actions = train_viz_data[f"E{i}"]["Action"]
                        taken_actions = train_viz_data[f"E{i}"]["Taken Action"]
                        len_actions = len(train_viz_data[f"E{i}"]["Action"])
                        print("Chosen actions in last episode:", "SELL -", str(round(actions.count(0)/len_actions, 2)), " / HOLD -", str(round(actions.count(1)/len_actions, 2)),  "/ BUY -", str(round(actions.count(2)/len_actions, 2)))
                        print("Taken actions in last episode: ", "SELL -", str(round(taken_actions.count(0)/len_actions, 2)), " / HOLD -", str(round(taken_actions.count(1)/len_actions, 2)),  "/ BUY -", str(round(taken_actions.count(2)/len_actions, 2)))
                    
                    # Update return curves & actions
                    learning_curves["Episode"].extend([i, i])
                    learning_curves["Return"].extend([sum(train_viz_data[f"E{i}"]["Reward"]), sum(val_viz_data["Reward"])])
                    learning_curves["Stage"].extend(["Training", "Validation"])

                    # terminate episode loop
                    break

                # train the network
                self.train(batch_size=batch_size)

                # update target network
                if i % target_update_freq:
                    self.update_target_network()

            # Cosmetics for plots
            train_viz_data[f"E{i}"]["Action"].append(None)
            train_viz_data[f"E{i}"]["Taken Action"].append(None)
        
        # Save policy returned at last episode
        policy = self.return_policy_Qvalues()

        return train_viz_data, val_viz_data, policy, learning_curves


    def evaluate(self, validation_env) -> Tuple:
        """
        Function to evaluate the network.
        :param validation_env: Validation environment
        :returns:
        """
        # Initialize the state
        state = validation_env.reset()
        
        # Keep track of visualisation data
        val_viz_data = defaultdict(List)
        val_viz_data = {
                "Waterlevel": [state[0]],
                "Price": [state[1]], 
                "Action": [],
                "Taken Action": [],
                "Reward": [0],
                "Shaped Reward": [0]}

        state = torch.tensor(state, dtype=torch.float)        

        while True:

            # normalisation with mean and std
            state[:1] = (state[:1] - self.env.water_mean) / self.env.water_std
            state[1:2] = (state[1:2] - self.env.price_mean) / self.env.price_std

            action = torch.argmax(self.online_network(state)).item()

            # next transition online
            new_state, reward, shaped_reward, done_online, _, taken_action = validation_env.step(action)


            # update viz data
            val_viz_data["Waterlevel"].append(new_state[0])
            val_viz_data["Price"].append(new_state[1])
            val_viz_data["Action"].append(action)
            val_viz_data["Taken Action"].append(taken_action)
            val_viz_data["Reward"].append(reward)
            val_viz_data["Shaped Reward"].append(shaped_reward)

            if done_online:
                break

            state = torch.tensor(new_state, dtype=torch.float)

        # Cosmetics for plots
        val_viz_data["Action"].append(None)
        val_viz_data["Taken Action"].append(None)

        return val_viz_data

    def evaluate_vincent(self, validation_env) -> Tuple:
        """
        Function to evaluate the network.
        :param validation_env: Validation environment
        :returns:
        """
        # Initialize the state
        state = validation_env.observation()[:2]

        # Keep track of visualisation data
        val_viz_data = defaultdict(List)
        val_viz_data = {
            "Waterlevel": [state[0]],
            "Price": [state[1]],
            "Action": [],
            "Reward": [0]}

        state = torch.tensor(state, dtype=torch.float)

        while True:

            # normalisation with mean and std
            state[:1] = (state[:1] - self.env.water_mean) / self.env.water_std
            state[1:2] = (state[1:2] - self.env.price_mean) / self.env.price_std

            action = torch.argmax(self.online_network(state)).item()

            # change actions to vincents actions
            vincent_action = 0
            if action == 0:
                vincent_action = -1
            elif action == 1:
                vincent_action = 0
            elif action == 2:
                vincent_action = 1

            # next transition online
            new_state, reward, terminated, truncated, _ = validation_env.step(vincent_action)

            # update viz data
            val_viz_data["Waterlevel"].append(new_state[0])
            val_viz_data["Price"].append(new_state[1])
            val_viz_data["Action"].append(action)
            val_viz_data["Reward"].append(reward)

            if terminated or truncated:
                break

            state = torch.tensor(new_state[:2], dtype=torch.float)

        # Cosmetics for plots
        val_viz_data["Action"].append(None)

        return val_viz_data
