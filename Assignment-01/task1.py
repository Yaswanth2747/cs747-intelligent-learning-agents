"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need

def kl(p, q):
    if q <= 0 or q >= 1:
        return float("inf")
    if p == 0:
        return math.log(1/(1-q))
    if p == 1:
        return math.log(1/q)
    return p * math.log(p/q) + (1-p) * math.log((1-p)/(1-q))
    

def solve_q(p, C, tol=1e-9, max_iter=100):
    lo, hi = p, 1 - 1e-9   # To ensure we look only to the right of p
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = kl(p, mid)
        if val > C:
            hi = mid
        else:
            lo = mid
        if abs(val - C) < tol:
            return mid
    return (lo + hi) / 2


"""
np.argmax will pick the first index if there are multiple entries that are maximum value in case of ties, 
so this is kind of like a little bit deterministic on breaking ties, so I have tried using
the below function to break ties uniformly randomly. Even though I am getting similar results,
I've stick to normal argmax for faster implementation for all 3 algorithms.
"""

# def argmax_random_tie(arr):
#     max_val = np.max(arr)
#     candidates = np.where(arr == max_val)[0]
#     return np.random.choice(candidates)


# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        
        self.counts = np.zeros(num_arms)        # To keep the count of how many times each arm is pulled, uta in our class notation
        self.values = np.zeros(num_arms)        # Contains the running empirical mean of each arm, pta in our class notation
        self.ucb_t  = np.zeros(num_arms)        # Contains the running UCB values for all the arms, gets activated when (start_ucb == True)
        self.start_ucb = False                  # At first, we pull each arm once, then we start the UCB based pulling
        self.start_arms = 0
        ## ucb_t is updated in get_reward
        ## Have to confirm whether no. of arms is greater than horizon or not in all test cases.
        
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        
        if (self.start_ucb):
            return np.argmax(self.ucb_t) ## argmax deterministically breaks ties, first index of the max val is returned
        else :
            self.start_arms = self.start_arms + 1 # each arm being pulled once before using UCB
            if (len(self.counts) == self.start_arms):
                self.start_ucb = True
            return self.start_arms-1
        
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        ## Computing running Empirical mean, same as given in epsilon-greedy
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        
        ## Updating upper confidence bound
        t = np.sum(self.counts) + 1 ## gives the current time
        for i in range(len(self.counts)):
            if self.counts[i] > 0:
                self.ucb_t[i] = self.values[i] + math.sqrt( ( 2 * math.log(t)) / self.counts[i] )
            else :
                self.ucb_t[i] = 0 
        
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        
        self.counts = np.zeros(num_arms)        # To keep the count of how many times each arm is pulled, uta in our class notation
        self.values = np.zeros(num_arms)        # Contains the running empirical mean of each arm, pta in our class notation
        self.ucb_t  = np.zeros(num_arms)        # Contains the running KL_UCB values for all the arms, gets activated when (start_ucb == True)
        self.start_ucb = False                  # At first, we pull each arm once, then we start the KL_UCB based pulling
        self.start_arms = 0
        
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        
        if (self.start_ucb):
            return np.argmax(self.ucb_t)
        else :
            self.start_arms = self.start_arms + 1 # each arm being pulled once before using UCB
            if (len(self.counts) == self.start_arms):
                self.start_ucb = True
            return self.start_arms-1
        
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        
        ## Updating the upper confidence bound
        t = int(np.sum(self.counts)) + 1   # current time
        l = math.log(t)
        c = 1
        for i in range(len(self.counts)):
            if self.counts[i] > 0:
                bound = (l + c * math.log(l)) / self.counts[i]
                self.ucb_t[i] = solve_q(self.values[i], bound)
            else :
                self.ucb_t[i] = 0 
        
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        
        # Each arm starts with a Beta(1,1) prior, so we don't need to pull each arm initially atleast once for fair exploration
        self.alphas = np.ones(num_arms)   # no. of successes + 1
        self.betas  = np.ones(num_arms)   # no. of failures + 1
        
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        
        # We sample from Beta distribution for each arm, and we be greedy on the sample, not the mean
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(self.num_arms)]
        return np.argmax(samples)
        
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        # we update Beta distribution for that arm only, others remain unchanged
        if reward == 1:
            self.alphas[arm_index] += 1
        else:
            self.betas[arm_index] += 1
        
        # END EDITING HERE