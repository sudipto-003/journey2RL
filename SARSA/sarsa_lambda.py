import numpy as np


class SARSA_Lam:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=0.1, lam=0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.q_a = np.zeros((self.state_space, self.action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam

    def init_elibility_trace(self):
        self.e_t = np.zeros((self.state_space, self.action_space))

    def choose_action(self, cur_state):
        trash_hold = np.random.uniform(0.0, 1.0)

        if trash_hold < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_a[cur_state])

        return action

    def update(self, cur_state, cur_action, reward, next_state, next_action):
        self.delta = reward + self.gamma * self.q_a[next_state, next_action] - self.q_a[cur_state, cur_action]
        self.e_t[cur_state, cur_action] += 1

        for s in range(self.state_space):
            for a in range(self.action_space):
                self.q_a[s, a] += self.alpha * self.delta * self.e_t[s, a]
                self.e_t[s, a] = self.gamma * self.lam * self.e_t[s, a]    