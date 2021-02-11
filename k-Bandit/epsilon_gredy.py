import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import seaborn as sns

class Bandit:
    def __init__(self, k=10, epsilon=0):
        self.i_rewards = np.array([[4, 2, 1, 3],
                                    [1, 10, 2, 0],
                                    [10, 7, 6, 4],
                                    [2, 4, 6, 8],
                                    [3, 2, 7, 0],
                                    [2, 8, 10, 1],
                                    [3, 9, 7, 0],
                                    [9, 1, 2, 0],
                                    [1, 8, 7, 0],
                                    [7, 2, 1, 8]])

        self.i_probability = np.array([[.03, .45, .25, .27],
                                        [.3, .5, .2, 0],
                                        [.05, .45, .35, .15],
                                        [.25, .25, .25, .25],
                                        [.75, .2, .05, 0],
                                        [.4, .4, .1, .1],
                                        [.25, .25, .5, 0],
                                        [.25, .5, .25, 0],
                                        [.2, .4, .4, 0],
                                        [.1, .1, .45, .35]])

        self.k = k
        self.epsilon = epsilon
        self.indices = np.arange(self.k)
        self.rng = default_rng()

    
    def new_run(self):
        self.q_true = self.rng.choice(10, 10)
        self.q_estimations = np.zeros(self.k)
        self.action_count = np.ones(self.k)


    def chose_action(self):
        toss = np.random.uniform(0.0, 1.0)
        if toss < self.epsilon:
            return self.rng.choice(self.indices)

        best_estimation = np.max(self.q_estimations)
        action = self.rng.choice(np.argwhere(self.q_estimations == best_estimation)[0])

        return action

    
    def reward_count(self, a_i):
        reward = self.rng.choice(self.i_rewards[a_i], p=self.i_probability[a_i])
        #reward = self.rng.integers(-2, 2) + self.q_true[a_i]
        self.q_estimations[a_i] += (reward - self.q_estimations[a_i]) / self.action_count[a_i]
        self.action_count[a_i] += 1

        return reward


def simulate_k_brandit(bandits, run, time):
    rewards = np.zeros((len(bandits), run, time))

    for i, bandit in enumerate(bandits):
        for j in range(run):
            bandit.new_run()
            for t in range(time):
                action_t = bandit.chose_action()
                reward_t = bandit.reward_count(action_t)
                rewards[i, j, t] = reward_t

    mean_rewards = np.mean(rewards, axis=1)
    return mean_rewards



if __name__ == '__main__':
    epsilons = [0.2, 0.1, 0.025, 1]
    bandits = [Bandit(epsilon=e) for e in epsilons]
    final_rewards = simulate_k_brandit(bandits, 100, 1000)
    print(final_rewards.shape)
    for ep, reward in zip(epsilons, final_rewards):
        plt.plot(reward, label=f'epsilon {ep}')
    plt.xlabel('steps')
    plt.ylabel('avg rewards')
    plt.legend()
    plt.show()
    '''
    b = Bandit()
    b.new_run()
    print(b.i_probability[1])
    svg = default_rng()
    n = svg.choice(b.i_rewards[0], p=b.i_probability[0])
    print(n)
    a = b.chose_action()
    print(a)
    r = b.reward_count(a)
    print(r)
    '''
    