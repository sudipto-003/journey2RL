import sys
sys.path.append('/home/sudipto/Reinforcement/DQN/')
from grid import Grid
from my_dqn import DQN
from replayQ import replayQ
import numpy as np
import time
import matplotlib.pyplot as plt


env = Grid()
action_space = env.action_space.n

dqn = DQN(action_space, hunits1=10, hunits2=10)
buffer = replayQ(10000)
batch_size = 32
cur_frame = 0
epsilon = 0.99
num_episodes = 1000
trewards = []
path = './checkpoints/grid_weights'

for episode in range(1, num_episodes+1):
    state = env.reset()
    ep_reward, done = 0, False
    while not done:
        state = state.flatten()
        state_in = np.expand_dims(state, axis=0)
        action = dqn.chose_action(state_in, epsilon)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        ep_reward += reward

        buffer.add(state, action, reward, next_state.flatten(), done)
        state = next_state
        cur_frame += 1

        if cur_frame % 500 == 0:
            dqn.copy_weights_target_network()

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            loss = dqn.train_dqn(states, actions, rewards, next_states, dones)

    if episode < 485:
        epsilon -= 0.002

    trewards.append(ep_reward)


dqn.dump_weights(path)
print(dqn.q_nn.layer1.weights)
env.close()

plt.plot(range(1, 1001), trewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('5x5GridWithDQN.jpg')
plt.show()