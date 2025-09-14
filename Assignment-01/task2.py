import numpy as np
from typing import List, Optional, Dict, Tuple

# =========================================================
# ===============   ENVIRONMENT (Poisson)   ===============
# =========================================================

class PoissonDoorsEnv:
    """
    Poisson environment with K doors.
    - Each door i has damage ~ Poisson(mu_i).
    - Each door starts with health H0, decreases as damage occurs.
    - Game ends when ANY door health < 0.
    """
    def __init__(self, mus: List[float], H0: int = 100, rng: Optional[np.random.Generator] = None):
        self.mus = np.array(mus, dtype=float)
        assert np.all(self.mus > 0), "Poisson means must be > 0"
        self.K = len(mus)
        self.H0 = H0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        """Reset healths to full H0."""
        self.health = np.full(self.K, self.H0, dtype=float)
        self.t = 0
        return self.health.copy()

    def step(self, arm: int) -> Tuple[float, bool, Dict]:
        """Pull a door (arm) → get Poisson damage, update health."""
        reward = float(self.rng.poisson(self.mus[arm]))
        self.health[arm] -= reward
        self.t += 1
        done = np.any(self.health < 0.0)
        return reward, done, {
            "reward": reward,
            "health": self.health.copy(),
            "t": self.t
        }


# =========================================================
# =====================   POLICIES   ======================
# =========================================================

class Policy:
    """
    Base Policy:

    here we track:
      - counts[i] = # times door i was probed
      - sums[i]   = total observed damage
      - Hcurr[i]  = current health
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None, H0: float = 100.0):

        self.K = K
        self.rng = rng if rng is not None else np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.sums   = np.zeros(K, dtype=float)
        self.H0 = H0
        self.Hcurr = np.full(self.K, self.H0, dtype=float) # this array holds the current health of all the doors
        # this is used by our select_arm func.

    def reset_stats(self):

        self.counts[:] = 0
        self.sums[:]   = 0.0
        self.Hcurr[:]  = self.H0

    def update(self, arm: int, reward: float):

        self.counts[arm] += 1
        self.sums[arm]   += reward
        self.Hcurr[arm] -= reward   # updating the current healths of all the doors

    @property
    def means(self) -> np.ndarray:
        """Return empirical mean damage for each door."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.sums / np.maximum(self.counts, 1)


# =========================================================
# ================   STUDENT POLICY (Task2)   =============
# =========================================================

class StudentPolicy(Policy):
    """
    Explore-then-commit:
      1. Probe each door once to estimate means.
      2. Compute expected_strikes = Hcurr / mean damage.
      3. Commit permanently to the door with MIN expected_strikes.
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None, H0: float = 100.0):
        super().__init__(K, rng, H0)
        self.best_arm: Optional[int] = None
        self.commit = False
    def reset_stats(self):
        """Reset and forget commitment."""
        super().reset_stats()
        self.best_arm = None
        self.commit = False

    def select_arm(self, t: int) -> int:
        
        #print(self.Hcurr)
        
        # If already committed → keep pulling best_arm
        if  self.commit == True:
            return self.best_arm

        for a in range(self.K):
        # Initial Exploration : we hit each door once initially like we did in task-1
            if self.counts[a] < 1:
                return a
        # After Commitment : Fixed exploitation from the first door whose health goes below 90
            if self.Hcurr[a] <= 90:
                self.best_arm = a
                self.commit = True
                return a

        # Commitment step : choose based on health-adjusted expected strikes
        means = self.means
        expected_strikes = np.full(self.K, np.inf) 
        np.divide(self.Hcurr, means, out=expected_strikes, where=means > 0) # H/mean

        # Before Commitment : We pick the door with least expected strikes before commitment.
        self.best_arm = int(np.argmin(expected_strikes))
        return self.best_arm

    def update(self, arm: int, reward: float):
        super().update(arm, reward)