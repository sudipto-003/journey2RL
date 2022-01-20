import gym
from ddpg import Agent
import matplotlib.pyplot as plt
import numpy as np

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)
agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], batch_size=128)
n_episodes = 250
history = []
critic_loss = []
actor_loss = []

for i in range(n_episodes):
    loss1 = []
    loss2 = []
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, _ = env.step(action)
        score += reward
        agent.remember(obs, action, reward, obs_, done)
        agent.learn()
        obs = obs_

    history.append(score)


avg = np.zeros(len(history))
for i in range(len(avg)):
    avg[i] = np.mean(history[max(0, i-25):(i+1)])
plt.plot(range(1, n_episodes+1), avg)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'{env_name} Showing Mean Reward')
plt.savefig(f'results/{env_name}-DDPG-Showing-Mean-Rewards-b128.jpg')
plt.show()
