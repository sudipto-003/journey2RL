import numpy as np

class Q_Agent:
    def __init__(self, state_space, action_space, gamma=0.99, alpha=0.3, epsilon=0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_a = np.zeros((self.state_space, self.action_space))

    def choose_action(self, cur_state):
        trash_hold = np.random.uniform(0.0, 1.0)
        if trash_hold < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_a[cur_state])

        return action

    def update(self, cur_state, cur_action, reward, next_state):
        self.q_a[cur_state, cur_action] += self.alpha * (reward + self.gamma * np.max(
                                            self.q_a[next_state]) - self.q_a[cur_state, cur_action])