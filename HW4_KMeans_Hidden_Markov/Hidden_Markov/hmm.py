from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        
        for j in range(S):
            alpha[j, 0] = self.pi[j] * self.B[j, self.obs_dict.get(Osequence[0])]
  
        for t in range(1, L):
            for j in range(S):
                alpha[j, t] = self.B[j, self.obs_dict.get(Osequence[t])] * sum([self.A[i, j] * alpha[i, t - 1] for i in range(S)])

  
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        
        for j in range(S):
            beta[j, L - 1] = 1
  
        for t in reversed(range(L - 1)):
            for i in range(S):
                beta[i, t] = sum([beta[j, t + 1] * self.A[i, j] * self.B[j, self.obs_dict.get(Osequence[t + 1])] for j in range(S)])
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        alpha = self.forward(Osequence)
        
        prob = sum(alpha[:, -1])
        
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        prob = 0
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        
        prob = (alpha * beta) / sum(alpha[:, -1])
        
        
        return prob

    # TODO:
    def viterbi(self, Osequence):
        
        S = len(self.pi)
        sig = [{}]
        path = {}     
        for y in range(S):
            sig[0][y] = self.pi[y] * self.B[y][self.obs_dict.get(Osequence[0])]
            path[y] = [y]
        for t in range(1, len(Osequence)):
            sig.append({})
            newpath = {}
            for y in range(S):
              (prob, state) = max((sig[t-1][y0] * self.A[y0][y] * self.B[y][self.obs_dict.get(Osequence[t])], y0) for y0 in range(S))
              sig[t][y] = prob
              newpath[y] = path[state] + [y]     
            path = newpath
        n = 0
        if len(Osequence)!=1:
          n = t
        (prob, state) = max((sig[n][y], y) for y in range(S))
        path = path[state]
      
        
        path1 = []
        
        for p in path:
            for key,value in self.state_dict.items():
                if value == p:
                    path1.append(key)
        
        return path1
