import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def generate_colors(n):
    colors = []
    while len(colors) < n:
        color = (random.random(), random.random(), random.random())
        if color not in colors:  # prevent exact repeats
            colors.append(color)
    return colors


def display_plot(rewards_list, episodes_list, names, plot_title, save=False):

    # Calculate running averages
    window_size = 100

    avg_rewards_list = []
    for rewards in rewards_list:
        avg = running_average(rewards, window_size)
        avg_rewards_list.append(avg)

    colors = generate_colors(len(avg_rewards_list))
    # Create the plot
    plt.figure(figsize=(12, 6))

    for index, avg in enumerate(avg_rewards_list):
        plt.plot(episodes_list[index][window_size-1:], avg, color=colors[index],
                 linewidth=2, label=f'Entropy Weight[{names[index]}] {window_size}-ep avg')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Comparison of {plot_title}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    if save:
        plt.savefig(f'Comparison of {plot_title}.png',
                    dpi=300, bbox_inches='tight')
        print("Plot saved")
    plt.show()


def save_reward(path, rewards):
    with open(path, 'w') as f:
        for reward in rewards:
            f.write(f"{reward}\n")


def save_model(path, actors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i, actor in enumerate(actors):
        torch.save(actor.state_dict(), f"{path}_agent{i}.pt")


def load_actors(actor_class, num_agents, input_size, output_size, path):
    actors = []
    for i in range(num_agents):
        actor = actor_class(input_size, output_size)
        actor.load_state_dict(torch.load(f"{path}_agent{i}.pt"))
        actor.eval()  # Optional: set to eval mode if only evaluating
        actors.append(actor)
    return actors
