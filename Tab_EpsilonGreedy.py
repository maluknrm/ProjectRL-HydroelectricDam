import numpy as np


class TabEpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """


    def sample_action(self, obs, epsilon, Q, state_dim):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        random_prob = np.random.uniform(0, 1)
        if random_prob < epsilon:
            action = np.random.randint(0, 3)

        else:
            state_action_row = self.make_space_specific(state_dim, Q, obs)
            action = np.argmax(state_action_row)

        return action


    def get_epsilon(self, iteration, epsilon_end):
        """
        Returns the new epsilon for the epsilon greedy policy.
        :param iteration: The current iteration.
        :param epsilon_end: The smallest possible epsilon.
        :returns the new epsilon
        """
        return 1 - (iteration * 0.00095) if iteration <= 1000 else epsilon_end


    def make_space_specific(self, state_dim, Q, obs):
        state_action_row = None

        if state_dim == 2:
            state_action_row = Q[obs[0], obs[1], ]
        
        elif state_dim == 3:
            state_action_row = Q[obs[0], obs[1], obs[2],]
        
        elif state_dim == 4:
            state_action_row = Q[obs[0], obs[1], obs[2], obs[3], ]

        return state_action_row




