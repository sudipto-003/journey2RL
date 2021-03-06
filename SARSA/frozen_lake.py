import gym
from sarsa_agent import SARSA
import os
import logging

def clear():
    os.system('clear')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('frozen-lake.log')
fh.setLevel(logging.INFO)
fh2 = logging.FileHandler('frozen-lake-goal.log')
fh2.setLevel(logging.ERROR)
logger.addHandler(fh)
logger.addHandler(fh2)

env = gym.make('FrozenLake8x8-v0')
action_space = env.action_space.n
state_space = env.observation_space.n
agent = SARSA(state_space, action_space, alpha=0.3, gamma=0.6)

for episode in range(1, 1001):
    cur_state = env.reset()
    cur_action = agent.choose_action(cur_state)
    step = 1
    t_reward = 0
    clear()
    print(f'================')
    print(f'||             ||')
    print(f'|| Episode: {episode}  ||')
    print(f'================')
    clear()
    env.render()
    while(True):
        next_state, reward, done, info = env.step(cur_action)
        t_reward += reward
        next_action = agent.choose_action(next_state)
        agent.update(cur_state, cur_action, reward, next_state, next_action)
        cur_state, cur_action = next_state, next_action
        clear()
        env.render()

        if done:
            if reward == 1:
                logger.error(f'Reached Goal at Episode {episode} total reward {reward}')
            break

        step += 1
    
    logger.info(f'Episode {episode} ended with total reward {t_reward}.')

env.close()
