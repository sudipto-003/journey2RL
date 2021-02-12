import numpy as np
from outside_environment import Environment
import math
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self):
        self._k = Environment.num_actions()
        self.action_indices = np.arange(self._k)
        self.rng = np.random.default_rng()

    @abstractmethod
    def refresh(self):
        pass

    @abstractmethod
    def chose_action(self):
        pass

    @abstractmethod
    def evaluate(self, ac_ind):
        pass



class GreedyAgent(Agent):
    def __init__(self, epsilon=0.025, init_rewards=0):
        Agent.__init__(self)
        self.epsilon = epsilon
        self.initial = init_rewards
        self.refresh()
        
    def refresh(self):
        self.q_estimates = np.zeros(self._k) + self.initial
        self.action_counts = np.zeros(self._k)

    def chose_action(self):
        toss = np.random.uniform(0.0, 1.0)
        if toss < self.epsilon:
            action_i = self.rng.choice(self.action_indices)

        else:
            best_estimation = np.max(self.q_estimates)
            action_i = self.rng.choice(np.argwhere(self.q_estimates == best_estimation)[0])

        return action_i

    def evaluate(self, ac_ind):
        true_reward = Environment.get_reward(ac_ind)
        self.action_counts[ac_ind] += 1
        self.q_estimates[ac_ind] += (true_reward - self.q_estimates[ac_ind]) / self.action_counts[ac_ind]

        return true_reward


class UCBAgent(Agent):
    def __init__(self, c=2):
        Agent.__init__(self)
        self.c = c
        self.refresh()

    def refresh(self):
        self.action_counts = np.zeros(self._k)
        self.time = 1
        self.estimates = np.zeros(self._k)

    def calculate_ucb(self, x, y):
        return x + self.c * math.sqrt(math.log(self.time) / (1e-9 + y))

    def chose_action(self):
        ucb_s = np.array([self.calculate_ucb(x, y) for x, y in zip(self.estimates, self.action_counts)])
        action_t = self.rng.choice(np.argwhere(ucb_s == max(ucb_s))[0])

        return action_t

    def evaluate(self, ac_ind):
        reward = Environment.get_reward(ac_ind)
        self.action_counts[ac_ind] += 1
        self.estimates[ac_ind] += (reward - self.estimates[ac_ind]) / self.action_counts[ac_ind]
        self.time += 1.0

        return reward


class GradientAgent(Agent):
    def __init__(self, alpha=0.1):
        Agent.__init__(self)
        self.alpha = alpha
        self.probabilites = np.zeros(self._k)
        self.average_reward = 0
        self.refresh()

    def refresh(self):
        self.preferences = np.zeros(self._k)
        self.time = 0

    def chose_action(self):
        self.apply_softmax_distribution()
        action_t = self.rng.choice(self.action_indices, p=self.probabilites)
        
        return action_t

    def apply_softmax_distribution(self):
        exponens = np.exp(self.preferences)
        self.probabilites = exponens / sum(exponens)

    def evaluate(self, ac_ind):
        reward = Environment.get_reward(ac_ind)
        self.preferences[ac_ind] += self.alpha * (reward - self.average_reward) * self.probabilites[ac_ind]
        for i in range(self._k):
            if i != ac_ind:
                self.preferences[i] -= self.alpha * (reward - self.average_reward) * self.probabilites[i]

        if self.time > 0:
            self.average_reward += (reward - self.average_reward) / self.time
        
        self.time += 1

        return reward