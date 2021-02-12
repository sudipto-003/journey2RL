import numpy as np
import matplotlib.pyplot as plt
#from epsilon_agent import Agent
#from ucb_agent import UCB_Agent
#from gradient_agent import GradientAgent
import math
from bandit_agents import GreedyAgent, UCBAgent, GradientAgent

def simulate_agents(agents, run, step):
    result = np.zeros((len(agents), run, step))
    for i, agent in enumerate(agents):
        for r in range(run):
            for t in range(step):
                t_action = agent.chose_action()
                t_reward = agent.evaluate(t_action)
                result[i, r, t] = t_reward
            agent.refresh()

    mean_rewards = np.mean(result, axis=1)
    return mean_rewards


def plot_reward(data, labels):
    for label, rel in zip(labels, data):
        plt.plot(rel, label=f'{label}')
    
    plt.xlabel(f'Steps')
    plt.ylabel(f'avg_values')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    initials = ['UCB c=sqrt(2)', 'epsilon=0.25', 'Gradient a=0.1']
    agents = []
    agents.append(UCBAgent(c=math.sqrt(2)))
    agents.append(GreedyAgent())
    agents.append(GradientAgent())
    data = simulate_agents(agents, 1000, 500)
    plot_reward(data, initials)
