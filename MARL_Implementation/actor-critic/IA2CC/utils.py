import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from matplotlib.animation import FFMpegWriter
import json
from environment import MultiAgentGridEnv
GRID_FILE = 'grid_world.json'


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
                 linewidth=2, label=f'{window_size}-ep avg')

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


# Save cumulative reward from each episode
def save_reward(path, rewards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for reward in rewards:
            f.write(f"{reward}\n")

# Visualize and save the trajectory of an episode


def visualize_trajectory(initial_positions, episode_actions, filename=None):
    print('Running visual')

    env = MultiAgentGridEnv(
        grid_file=GRID_FILE,
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4,
        initial_positions=initial_positions
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory != '':
            os.makedirs(directory, exist_ok=True)

        writer = FFMpegWriter(fps=2)
        with writer.saving(fig, filename, dpi=100):
            # Capture the initial state
            ax.clear()
            env.render(ax, actions=None, step=0)
            writer.grab_frame()
            plt.pause(0.1)

            for step, actions in enumerate(episode_actions, start=1):
                env.step(actions)
                ax.clear()
                env.render(ax, actions=actions, step=step)
                writer.grab_frame()
                plt.pause(0.1)
        print(f"Visualization saved as {filename}")
    else:
        ax.clear()
        env.render(ax, actions=None, step=0)
        plt.pause(0.5)

        for step, actions in enumerate(episode_actions, start=1):
            env.step(actions)
            ax.clear()
            env.render(ax, actions=actions, step=step)
            plt.pause(0.5)

    plt.close(fig)


# Save metrics in a CSV file
def save_metrics(metrics, filename):
    directory = os.path.dirname(filename)
    if directory != '':
        os.makedirs(directory, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(list(metrics.items()),
                      columns=['Metric', 'Value'])

    # Save to CSV
    df.to_csv(filename, index=False)

    print(df)


# Evluate on unseen seeds
def evaluate(model, episode_count=50):

    for episode in range(50, episode_count+50):
        run_episode(episode, model)


# Test on one single seed
def run_episode(seed, model):
    env = MultiAgentGridEnv(
        grid_file=GRID_FILE,
        coverage_radius=4,
        max_steps_per_episode=200,
        num_agents=4,
        seed=seed
    )

    initial_positons = env.initial_positions
    episode_actions = []

    obs, _ = env.reset()
    done = False
    while not done:
        # Get sensor readings
        sensor_readings = env.get_sensor_readings()

        # Forward pass
        actions, _, _ = model.act(obs, sensor_readings)

        # Step
        obs, _, done, _, _ = env.step(actions)

        # Record actions
        episode_actions.append(actions)

    # Save statistics
    metrics = env.get_metrics()
    save_metrics(metrics, f'evaluate/seed_{seed}/statistics.csv')

    # Save the trajectory
    visualize_trajectory(
        initial_positons, episode_actions, f'evaluate/seed_{seed}/trajectory.mp4')
