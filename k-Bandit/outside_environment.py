import numpy as np


class Environment:
    i_rewards = np.array([[4, 2, 1, 3],
                        [1, 10, 2, 0],
                        [10, 7, 6, 4],
                        [2, 4, 6, 8],
                        [3, 2, 7, 0],
                        [2, 8, 10, 1],
                        [3, 9, 7, 0],
                        [9, 1, 2, 0],
                        [1, 8, 7, 0],
                        [7, 2, 1, 8]])

    i_probability = np.array([[.03, .45, .25, .27],
                            [.3, .5, .2, 0],
                            [.05, .45, .35, .15],
                            [.25, .25, .25, .25],
                            [.75, .2, .05, 0],
                            [.4, .4, .1, .1],
                            [.25, .25, .5, 0],
                            [.25, .5, .25, 0],
                            [.2, .4, .4, 0],
                            [.1, .1, .45, .35]])

    rng = np.random.default_rng()

    @classmethod
    def get_reward(cls, action):
        reward = cls.rng.choice(cls.i_rewards[action], p=cls.i_probability[action])

        return reward

    @classmethod
    def num_actions(cls):
        return len(cls.i_rewards)