import matplotlib.pyplot as plt
import numpy as np
import random


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
    window_size = 50

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
