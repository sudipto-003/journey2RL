import gym
from sarsa_agent import SARSA
import os
import time

def clear():
    os.system('clear')

env = gym.make('FrozenLake-v0')
action_space = env.action_space.n
state_space = env.observation_space.n
agent = SARSA(state_space, action_space, state_space-1)

episode = 1
while(True):
    cur_state = env.reset()
    cur_action = agent.choose_action(cur_state)
    step = 1
    clear()
    print(f'================')
    print(f'||             ||')
    print(f'|| Episode: {episode}  ||')
    print(f'================')
    time.sleep(1)
    clear()
    env.render()
    time.sleep(0.5)
    while(True):
        next_state, reward, done, info = env.step(cur_action)
        next_action = agent.choose_action(next_state)
        agent.update(cur_state, cur_action, reward, next_state, next_action)
        cur_state, cur_action = next_state, next_action
        clear()
        env.render()
        time.sleep(0.5)
        
        if done:
            break

        step += 1
    
    #print(f'Episode {episode} terminated after {step} steps at state {cur_state}.')
    episode += 1
    if cur_state == state_space - 1:
        sep = '+'*37 if episode > 10 else '+'*36
        print(sep)
        print(f'+++Hurray we reached at episode {episode}+++')
        print(sep)
        break

env.close()
