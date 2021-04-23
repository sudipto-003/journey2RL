import numpy as np
import matplotlib.pyplot as plt
import argparse
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
    parser = argparse.ArgumentParser(description="Different exploration algorithms comparision with their hyper parameter")
    parser.add_argument('--run', '-r', type=int, required=True)
    parser.add_argument('--step', '-t', type=int, required=True)
    parser.add_argument('--ucb', '-u', action='extend', nargs='+', type=float)
    parser.add_argument('--epsilon', '-e', action='extend', nargs='+', type=float)
    parser.add_argument('--alpha', '-a', action='extend', nargs='+', type=float)
    parser.add_argument('--sqrtC', action='store_true')
    args = parser.parse_args()
    if args.sqrtC:
        args.ucb = [math.sqrt(c) for c in args.ucb]
    
    agents = []
    agent_labels = []
    if args.epsilon:
        for e in args.epsilon:
            agents.append(GreedyAgent(epsilon=e))
            agent_labels.append(f'Epsilon Greedy(e={e})')
    
    if args.ucb:
        for c in args.ucb:
            agents.append(UCBAgent(c=c))
            agent_labels.append(f'UCB(c={c})')
    
    if args.alpha:
        for a in args.alpha:
            agents.append(GradientAgent(alpha=a))
            agent_labels.append(f'Graditent with baseline(a={a})')

    data = simulate_agents(agents, args.run, args.step)
    plot_reward(data, agent_labels)
