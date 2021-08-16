import gym
from gym import spaces
import numpy as np
import sys, os


class Grid(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(4, )
        self.observation_shape = (5, 5)
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_shape),
            high=np.ones(self.observation_shape),
            dtype=np.float32
        )


    def reset(self):
        self.observation = np.zeros(self.observation_shape)
        rng = np.random.default_rng()
        row, column = rng.integers(5, size=2)
        if row == self.observation_shape[0]-1 and column == self.observation_shape[1]-1:
            salt1, salt2 = rng.integers(1, 5, size=2)
            row, column = row-salt1, column-salt2
        self.state = [row, column]
        self.observation[row, column] = 1

        return self.observation

    def step(self, action):
        done = False
        reward = -1

        if action == 0:
            new_state = [self.state[0]-1, self.state[1]]
        elif action == 1:
            new_state = [self.state[0]+1, self.state[1]]
        elif action == 2:
            new_state = [self.state[0], self.state[1]-1]
        elif action == 3:
            new_state = [self.state[0], self.state[1]+1]

        if self.out_of_boundary(new_state):
            reward = -5
        else:
            self.observation[self.state[0], self.state[1]] = 0
            self.state = new_state
            self.observation[self.state[0], self.state[1]] = 1

        if self.goal_reached(self.state):
            reward = 10
            done = True

        return self.observation, reward, done, {'state': self.state} 

    def out_of_boundary(self, state):
        max_row, max_column = self.observation_shape

        if state[0] < 0 or state[0] >= max_row or state[1] < 0 or state[1] >= max_column:
            return True
        
        return False

    def goal_reached(self, state):
        x, y = self.observation_shape

        if state[0] == x-1 and state[1] == y-1:
            return True

        return False

    def render(self, mode='human', close=False):
        os.system('clear')
        for i in range(self.observation_shape[0]):
            for j in range(self.observation_shape[1]):
                if i == self.state[0] and j == self.state[1]:
                    print(f'\033[91m{int(self.observation[i, j])}\033[00m', end=' '*2)
                else:
                    print(f'{int(self.observation[i, j])}', end=' '*2)
            print(end='\n')

        sys.stdout.flush()
