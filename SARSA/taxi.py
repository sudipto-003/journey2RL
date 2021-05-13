import gym
from sarsa_agent import SARSA
from sarsa_lambda import SARSA_Lam
import os
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('taxi-logs.log')
fh.setLevel(logging.INFO)
ch = logging.FileHandler('taxi-success.log')
ch.setLevel(logging.ERROR)
logger.addHandler(fh)
logger.addHandler(ch)

def clear():
    os.system('clear')

env = gym.make('Taxi-v3')
action_space = env.action_space.n
state_space = env.observation_space.n
agent = SARSA_Lam(state_space, action_space)

episode = 1
while episode <= 1000:
    agent.init_elibility_trace()
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
        next_action = agent.choose_action(next_state)
        agent.update(cur_state, cur_action, reward, next_state, next_action)
        cur_state, cur_action = next_state, next_action
        t_reward += reward
        clear()
        env.render()
        
        if done:
            if reward > 0:
                logger.error(f'Success gained reward {reward} at Episode {episode}')
            break

        step += 1
    
    logger.info(f'Episode {episode} ended with total Reward {t_reward}')
    episode += 1

env.close()