import sys
sys.path.append('/home/sudipto/Reinforcement/DQN/')
from grid import Grid
from my_dqn import DQN
import numpy as np
import matplotlib.pyplot as plt

env = Grid()
action_space = env.action_space.n

dqn = DQN(action_space, hunits1=10, hunits2=10)
cur_frame = 0
epsilon = 0.02
num_episodes = 200
path = './checkpoints/grid_weights'
state = env.reset()
state = state.flatten()
state_in = np.expand_dims(state, axis=0)
action = dqn.chose_action(state_in, epsilon)

status = dqn.reload_weights(path)
trewards = []
# print(dqn.q_nn.summary())
# print(dqn.q_nn.layer1.weights)
# status.assert_consumed()

for episode in range(1, num_episodes+1):
    state = env.reset()
    ep_reward, done = 0, False
    while not done:
        state = state.flatten()
        state_in = np.expand_dims(state, axis=0)
        action = dqn.chose_action(state_in, epsilon)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward

    trewards.append(ep_reward)

env.close()

plt.plot(range(1, num_episodes+1), trewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('5x5GridWithPretrainedDQN.jpg')
plt.show()