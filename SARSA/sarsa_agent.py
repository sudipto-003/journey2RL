import numpy as np
import math
import random

class SARSA:
    def __init__(self, S, A, terminate_state, gamma=0.9, epsilon=0.1):
        self.state_space = S
        self.action_space = A
        self.q_a = np.ones((S, A))
        self.q_a[terminate_state, :] = 0.0
        self.visits = np.ones(self.state_space)
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        toss = np.random.uniform(0.0, 1.0)
        if toss < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_a[state])

        return action

    def update(self, state, action, reward, next_state, next_action):
        self.q_a[state, action] += (1 / self.visits[state] += 1) * (reward + self.gamma * 
                                    self.q_a[next_state, next_action] - self.q_a[state, action])
        
        self.visits[state] += 1