"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------
class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed

def kl(p, q):
    p = min(max(p, 1e-12), 1-1e-12)
    q = min(max(q, 1e-12), 1-1e-12)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def solve_q(p, C, tol=1e-4, max_iter=20):
    lo, hi = p, 1 - 1e-12
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = kl(p, mid)
        if val > C:
            hi = mid
        else:
            lo = mid
        if abs(val - C) < tol:
            break
    return (lo + hi) / 2


# ------------------ Optimized KL-UCB Algorithm ------------------
class KL_UCB_Optimized:
    def __init__(self, num_arms, horizon):
        """
        Key Ideas : 
        - Use batched updates for KL-UCB, instead of updatng at each step in horizon
        - Also made the bound alot more tighter.
        - Removed extra ln(ln(t))
        """
        # almost same arrays and vars from Task-1
        self.num_arms = num_arms
        self.horizon = horizon
        self.counts = np.zeros(num_arms, dtype=int)
        self.values = np.zeros(num_arms, dtype=float)
        self.ucb_t = np.zeros(num_arms, dtype=float)
        self.start_ucb = False
        self.start_arms = 0
        self.total_steps = 0
        # this is new
        self.p = 6  # initial batch size

    def give_pull(self):
        # We Pull each arm twice initially
        # Reason : Since we are using a batch update type pulling strategy, we need fair amount of exploration initially,
        # Since we already have p = 6, for t < 100, it makes sense.
        if not self.start_ucb:
            arm = self.start_arms % 2
            self.start_arms += 1
            if self.start_arms >= 2*self.num_arms:
                self.start_ucb = True
            return arm

        # kl-ucb
        if self.total_steps % self.p == 0:
            t = np.sum(self.counts) + 1
            for i in range(self.num_arms):
                if self.counts[i] > 0:
                    # here I intentionally removed ln(ln(t)) term to reduce computational time and also to put a tighter bound,
                    # that's why there is that 0.1 factor
                    bound = (math.log(t)*0.1) / self.counts[i]
                    self.ucb_t[i] = solve_q(self.values[i], bound)
                else:
                    self.ucb_t[i] = float('inf')  # force exploration

            # Updating the batch size: first 100 steps p=6, then p = 55 from there on
            if self.total_steps >= 100:
                self.p = 55

        self.total_steps += 1
        return int(np.argmax(self.ucb_t))

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        self.values[arm_index] = ((n - 1) * self.values[arm_index] + reward) / n


# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE