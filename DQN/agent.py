import sys
sys.path.append('/home/sudipto/Reinforcement/MemoryBuffer/')
import gym
from my_dqn import DQN
from exp_replay import ExpReplay
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

env_name = 'MountainCar-v0'
env = gym.make(env_name)
dump_path = './checkpoints/cartpole'
# recorder = None
# recorder = VideoRecorder(env, f'/home/sudipto/Reinforcement/Videos/{env_name}-b256-h32-lr3e-3.mp4', enabled=True)
action_space = env.action_space.n

dqn = DQN(action_space, hunits1=16, hunits2=16)
replay = ExpReplay(100000)
num_episodes = 1000
batch_size = 128
cur_frame = 0
epsilon = 0.99
window = []
mean_rewards = []

# For training the agent
for episode in range(1, num_episodes+1):
    state = env.reset()
    ep_reward, done = 0, False
    while not done:
        #env.render()
        state_in = tf.expand_dims(state, axis=0)
        action = dqn.chose_action(state_in, epsilon)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward

        replay.add(state, action, reward, next_state, done)
        state = next_state
        cur_frame += 1

        if cur_frame % 500 == 0:
            dqn.copy_weights_target_network()

        if len(replay) >= batch_size:
            states, actions, rewards, next_states, dones = replay.sample(batch_size)
            loss = dqn.train_dqn(states, actions, rewards, next_states, dones)

    if episode < 485:
        epsilon -= 0.002

    if len(window) == 25:
        window = window[1:]
    window.append(ep_reward)
    mean_rewards.append(np.mean(window))

    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon {epsilon:.3f} '
            f'Mean Rewards in last 100 episode {mean_rewards[episode-1]:.3f}')

# Recording the performance of the trained agent
# test = 10
# for t in range(test):
#     state = env.reset()
#     ep_reward, done = 0, False
#     while not done:
#         #env.render()
#         recorder.capture_frame()
#         state_in = tf.expand_dims(state, axis=0)
#         action = dqn.chose_action(state_in, epsilon)
#         next_state, reward, done, _ = env.step(action)
#         ep_reward += reward

#         state = next_state

#     print(f'After Trained ad episode {t+1} Earned Reward {ep_reward}')


# recorder.close()
# recorder.enabled = False
# dqn.dump_weights(dump_path)
env.close()

plt.plot(range(1, 1001), mean_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'{env_name} Showing Mean Reward(Window=25)')
plt.savefig(f'{env_name}-DQN-Training-Mean-Rewards-hu16-b128-lr3e-3.jpg')
plt.show()
