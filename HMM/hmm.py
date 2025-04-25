from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################

        # Initialize the first column of alpha
        for s in range(S):
            alpha[s, 0] = self.pi[s] * self.B[s, O[0]]

        # Compute alpha for the rest of the sequence
        for t in range(1, L):
            for s in range(S):
                # Sum over all possible previous states
                alpha[s, t] = self.B[s, O[t]] * \
                    sum(alpha[s_prev, t-1] * self.A[s_prev, s]
                        for s_prev in range(S))

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################

        # Initialize the last column of beta with 1's
        for s in range(S):
            beta[s, L-1] = 1.0

        # Compute beta for the rest of the sequence in reverse order
        for t in range(L-2, -1, -1):
            for s in range(S):
                # Sum over all possible next states
                beta[s, t] = sum(self.A[s, s_next] * self.B[s_next, O[t+1]]
                                 * beta[s_next, t+1] for s_next in range(S))

        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################

        alpha = self.forward(Osequence)
        # The probability of the entire sequence is the sum of the final alpha values
        return np.sum(alpha[:, -1])

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
                           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################

        S = len(self.pi)
        L = len(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        gamma = np.zeros([S, L])

        # Compute gamma for each time step
        for t in range(L):
            # Normalization factor (probability of the entire sequence)
            norm_factor = np.sum(alpha[:, t] * beta[:, t])

            for s in range(S):
                # Posterior probability of being in state s at time t
                gamma[s, t] = (alpha[s, t] * beta[s, t]) / norm_factor

        return gamma

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        O = self.find_item(Osequence)

        # Compute joint probability of adjacent states for each pair of time steps
        for t in range(L-1):
            # Normalization factor (probability of the entire sequence)
            norm_factor = self.sequence_prob(Osequence)

            for i in range(S):
                for j in range(S):
                    # P(Z_t = i, Z_{t+1} = j | X_{1:T})
                    prob[i, j, t] = (alpha[i, t] * self.A[i, j] *
                                     self.B[j, O[t+1]] * beta[j, t+1]) / norm_factor

        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################

        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)

        # Initialize the delta and psi matrices
        delta = np.zeros((S, L))
        psi = np.zeros((S, L), dtype=int)

        # Initialize the first column of delta
        for s in range(S):
            delta[s, 0] = self.pi[s] * self.B[s, O[0]]
            psi[s, 0] = 0  # No previous state for t=1

        # Compute delta and psi for the rest of the sequence
        for t in range(1, L):
            for s in range(S):
                # Find the most likely previous state
                temp = delta[:, t-1] * self.A[:, s]
                psi[s, t] = np.argmax(temp)
                delta[s, t] = self.B[s, O[t]] * np.max(temp)

        # Backtracking to find the most likely state sequence
        # First, find the most likely final state
        best_path_indices = np.zeros(L, dtype=int)
        best_path_indices[L-1] = np.argmax(delta[:, L-1])

        # Then, backtrack to find the rest of the states
        for t in range(L-2, -1, -1):
            best_path_indices[t] = psi[best_path_indices[t+1], t+1]

        # Convert indices to actual state names
        for t in range(L):
            state_idx = best_path_indices[t]
            state_name = self.find_key(self.state_dict, state_idx)
            path.append(state_name)

        return path

    # DO NOT MODIFY CODE BELOW

    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
