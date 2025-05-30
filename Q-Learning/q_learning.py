"""
This module contains the implementation of a simple Q-learning agent with experience replay.

Instruction: Implement the missing parts. Only modify this file where it says "YOUR CODE HERE".
"""
import numpy as np
from finite_mdp import DiscreteSpace
from replay_buffer import ReplayBuffer
from utils import Transition


class QLearningAgent:
    def __init__(
        self, state_space: DiscreteSpace, action_space: DiscreteSpace, lr: float = 0.1,
        discount: float = 0.99, explore_rate: float = 0.1, buffer_capacity: int = 1,
        batch_size: int = 32
    ) -> None:
        """
        Initialize the Q-learning agent with the given parameters.

        Args:
            - state_space (DiscreteSpace): State space of the environment.
            - action_space (DiscreteSpace): Action space of the environment.
            - lr (float): Learning rate.
            - discount (float): Discount factor.
            - explore_rate (float): Exploration rate for epsilon-greedy policy.
            - buffer_capacity (int): Capacity of the replay buffer. 
            - batch_size (int): Batch size for learning from the replay buffer.
        """
        # DO NOT MODIFY
        self.lr = lr
        self.discount = discount
        self.explore_rate = explore_rate
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size

        self.state_space = state_space
        self.action_space = action_space

        self.q_table = None
        self.reset()
        # DO NOT MODIFY

    def reset(self) -> None:
        """
        Reset the agent's Q-table. 
        Note: 
            - The Q-table (shape: n_states x n_actions) should be initialized with IID standard normal.
        Hint: 
            - use np.random
        """
        ## YOUR CODE HERE ###
        # Initialize with standard normal distribution (as specified)
        self.q_table = np.random.standard_normal(
            size=(self.state_space.n, self.action_space.n)
        )
        ## END OF YOUR CODE ###

        # reset the reply buffer.
        self.replay_buffer.clear()

    def act(self, state: int, exploit: bool = False) -> int:
        """
        Select an action for the given state. Act according to greedy if `exploit = True`; 
        otherwise, use the epsilon-greedy exploration strategy (using `self.explore_rate`). 

        Args:
            - state: The current state.
            - exploit: Boolean flag for whether to exploit or not.

        Returns:
            The selected action (int).
        """
        ### YOUR CODE HERE ###
        if exploit or np.random.random() > self.explore_rate:
            # Exploit: choose action with highest Q-value
            # In case of ties, randomly select among actions with max Q-value
            max_q_value = np.max(self.q_table[state])
            max_actions = np.where(self.q_table[state] == max_q_value)[0]
            return np.random.choice(max_actions)
        else:
            # Explore: choose a random action
            return self.action_space.sample()
        ### END OF YOUR CODE ###

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Observe the data and store into the replay buffer (as a `Transition`).

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
        """
        ### YOUR CODE HERE ###
        transition = Transition(state, action, reward, next_state)
        self.replay_buffer.push(transition)
        ### END OF YOUR CODE ###

    def learn(self) -> dict:
        """
        Update the Q-table using a batch of transitions from the replay buffer.

        Returns:
            Dict summarizing the current learning step, including:
                - 'td_error': The TD error of the current learning step.
                - 'q_table': The updated Q-table.
                - 'value_arr': The array-representation of the value function.
                - 'policy_arr': The array-representation of the policy.

        Note:
            - Try to vectorize your code instead of simply looping over all samples in the batch.
            - `td_error` should have shape (batch_size,) i.e., one TD error per sampled transition.
        """
        batch = self.replay_buffer.sample(self.batch_size)
        ### YOUR CODE HERE ###
       # If batch is empty, return with zeros for td_error
        if batch.size() == 0:
            return {
                'td_error': np.array([]),
                'q_table': self.q_table,
                'value_arr': self.get_value_arr(),
                'policy_arr': self.get_policy_arr(),
            }

        # Vectorized implementation for better performance
        # Calculate current Q-values for the state-action pairs
        current_q_values = self.q_table[batch.states, batch.actions]
        
        # Calculate max Q-values for next states
        next_max_q_values = np.max(self.q_table[batch.next_states], axis=1)
        
        # Calculate target Q-values
        target_q_values = batch.rewards + self.discount * next_max_q_values
        
        # Calculate TD errors
        td_error = target_q_values - current_q_values
        
        # Update Q-table entries
        for i in range(batch.size()):
            self.q_table[batch.states[i], batch.actions[i]] += self.lr * td_error[i]

        ### END OF YOUR CODE ###
        return {
            'td_error': td_error,
            'q_table': self.q_table,
            'value_arr': self.get_value_arr(),
            'policy_arr': self.get_policy_arr(),
        }  # DO NOT MODIFY THE RETURN VALUE

    def get_policy_arr(self) -> np.ndarray:
        """
        Get the array-representation of the greedy policy from Q-table. 

        Returns:
            A 1D array where the i-th element is the action to be taken in the 
            i-th state according to the policy derived from the Q-table.
        """
        ### YOUR CODE HERE ###
        # For each state, pick the action with the highest Q-value
        return np.argmax(self.q_table, axis=1)
        ### END OF YOUR CODE ###

    def get_value_arr(self) -> np.ndarray:
        """
        Get the array-representation of the fitted value function from Q-table. The fitted value 
        function gives the estimated expected cumulative reward for each state following the current policy.

        Returns:
            A 1D array where each element represents the value of the corresponding state.
        """
        ### YOUR CODE HERE ###
        # For each state, the value is the maximum Q-value across all actions
        return np.max(self.q_table, axis=1)
        ### END OF YOUR CODE ###
