import gym
from my_dqn import DQN
from replayQ import replayQ
import tensorflow as tf
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

env_name = 'MountainCar-v0'
env = gym.make(env_name)
recorder = None
recorder = VideoRecorder(env, f'/home/sudipto/Reinforcement/Videos/{env_name}.mp4', enabled=True)
action_space = env.action_space.n

dqn = DQN(action_space, hunits1=32, hunits2=32)
replay = replayQ(100000)
num_episodes = 700
batch_size = 128
cur_frame = 0
epsilon = 0.99
last_100_rewards = []

# For training the agent
for episode in range(num_episodes+1):
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

        if cur_frame % 2000 == 0:
            dqn.copy_weights_target_network()

        if len(replay) >= batch_size:
            states, actions, rewards, next_states, dones = replay.sample(batch_size)
            loss = dqn.train_dqn(states, actions, rewards, next_states, dones)

    if episode < 485:
        epsilon -= 0.002

    if len(last_100_rewards) == 100:
        last_100_rewards = last_100_rewards[1:]
    last_100_rewards.append(ep_reward)

    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon {epsilon:.3f} '
            f'Mean Rewards in last 100 episode {np.mean(last_100_rewards):.3f}')

# Recording the performance of the trained agent
test = 10
for t in range(test):
    state = env.reset()
    ep_reward, done = 0, False
    while not done:
        #env.render()
        recorder.capture_frame()
        state_in = tf.expand_dims(state, axis=0)
        action = dqn.chose_action(state_in, epsilon)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward

        state = next_state

    print(f'After Trained ad episode {t+1} Earned Reward {ep_reward}')


recorder.close()
recorder.enabled = False
env.close()
