import gym
from Q import Q_Agent
import os
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('taxi-logs_q.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

def clear():
    os.system('clear')

env = gym.make('Taxi-v3')
action_space = env.action_space.n
state_space = env.observation_space.n
agent = Q_Agent(state_space, action_space)

episode = 1
while episode <= 1000:
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
        agent.update(cur_state, cur_action, reward, next_state)
        next_action = agent.choose_action(next_state)
        cur_state, cur_action = next_state, next_action
        t_reward += reward
        clear()
        env.render()
        
        if done:
            break

        step += 1
    
    logger.info(f'Episode {episode} ended with total Reward {t_reward}')
    episode += 1

#after tarining, let's see who the agent working.
episode = 1
while episode <= 100:
    cur_state = env.reset()
    cur_action = agent.choose_action(cur_state)
    step = 1
    t_reward = 0
    clear()
    print(f'================')
    print(f'||             ||')
    print(f'|| Test Episode: {episode}  ||')
    print(f'================')
    clear()
    env.render()
    time.sleep(0.1)
    while(True):
        next_state, reward, done, info = env.step(cur_action)
        agent.update(cur_state, cur_action, reward, next_state)
        next_action = agent.choose_action(next_state)
        cur_state, cur_action = next_state, next_action
        t_reward += reward
        clear()
        env.render()
        time.sleep(0.1)
        
        if done:
            break

        step += 1
    
    #logger.info(f'Episode {episode} ended with total Reward {t_reward}')
    episode += 1

env.close()