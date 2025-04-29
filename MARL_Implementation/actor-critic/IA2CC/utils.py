import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from matplotlib.animation import FFMpegWriter
import json
from environment import MultiAgentGridEnv
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
GRID_FILE = 'grid_world2.json'


def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def generate_colors():
    base_colors = [
        (0.121, 0.466, 0.705),  # blue
        (1.0, 0.498, 0.054),    # orange
        (0.172, 0.627, 0.172),  # green
        (0.839, 0.153, 0.157),  # red
        (0.580, 0.404, 0.741),  # purple
        (0.549, 0.337, 0.294),  # brown
        (0.890, 0.467, 0.761),  # pink
        (0.498, 0.498, 0.498),  # grey
        (0.737, 0.741, 0.133),  # yellow-green
        (0.090, 0.745, 0.811),  # cyan
    ]
    return base_colors


def display_plot(rewards_list, episodes_list, names, plot_title, filename='test', save=False):

    # Calculate running averages
    window_size = 100

    avg_rewards_list = []
    for rewards in rewards_list:
        avg = running_average(rewards, window_size)
        avg_rewards_list.append(avg)

    colors = generate_colors()
    # Create the plot
    plt.figure(figsize=(12, 6))

    for index, avg in enumerate(avg_rewards_list):
        plt.plot(episodes_list[index][window_size-1:], avg, color=colors[index],
                 linewidth=2, label=f'{names[index]} {window_size}-ep avg')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    if save:
        plt.savefig(filename,
                    dpi=300, bbox_inches='tight')
        print("Plot saved")
    plt.show()


def load_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data


def save_model_stats(name, model, env, max_episode, time, filename):
    training_stats = {}
    training_stats["Name"] = name
    training_stats["Actor"] = {
        "Learning rate": model.actor_learning_rate,
        "Input size": model.actor_input_size,
        "Output size": model.actor_output_size,
    }
    training_stats["Critic"] = {
        "Learning rate": model.critic_learning_rate,
        "Input size": model.critic_input_size,
    }

    training_stats["Discount Factor"] = model.gamma
    training_stats["Entropy Weight"] = model.entropy_weight
    training_stats["Number of Episodes"] = max_episode
    training_stats["Maximum Step Per Episode"] = env.max_steps_per_episode
    training_stats["Time taken"] = time

    training_stats["Reward Weight"] = env.reward_weight

    directory = os.path.dirname(filename)
    if directory != '':
        os.makedirs(directory, exist_ok=True)

    try:
        with open(filename, 'w') as fp:
            json.dump(training_stats, fp, indent=4)
    except Exception as e:
        print(f"Error saving file: {e}")

# Save cumulative reward from each episode


def save_reward(path, rewards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for reward in rewards:
            f.write(f"{reward}\n")

# Visualize and save the trajectory of an episode


def visualize_trajectory(initial_positions, episode_actions, filename=None):

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

            for step, actions in enumerate(episode_actions, start=1):
                env.step(actions)
                ax.clear()
                env.render(ax, actions=actions, step=step)
                writer.grab_frame()

        print(f"Visualization saved as {filename}")
    else:
        print("Filename not provided, skipping visualization saving.")

    plt.close(fig)


def save_evalutation_stats(env, metric, model_stats, filename):
    info = {}

    info['Metic'] = metric
    # Grid into
    info['Grid Info'] = {
        'Height': env.grid_height,
        'Width': env.grid_width,
        'Total Available Cells': int(env.total_cells_to_cover),
        'Number of UAVs': env.num_agents
    }
    info['UAV'] = {
        'Number of valid actions': env.get_total_actions(),
        'Local observation size': env.get_obs_size(),
        'Initial Positions': env.initial_positions,
        'Coverage Radius': env.coverage_radius
    }
    info['Max step allowed'] = env.max_steps_per_episode
    info['Seed'] = env.seed
    info['Reward Weight'] = model_stats['Reward Weight']

    directory = os.path.dirname(filename)
    if directory != '':
        os.makedirs(directory, exist_ok=True)

    try:
        with open(filename, 'w') as fp:
            json.dump(info, fp, indent=4)
    except Exception as e:
        print(f"Error saving file: {e}")


# Evluate on unseen seeds

def evaluate(model, model_stats,  episode_count=5, path=''):

    for episode in range(0, episode_count):
        run_episode(episode, model, model_stats, path=path)
        print('-----------------------------------\n')


# Test on one single seed
def run_episode(seed, model, model_stats, path=''):
    env = MultiAgentGridEnv(
        grid_file=GRID_FILE,
        coverage_radius=4,
        max_steps_per_episode=200,
        num_agents=4,
        seed=seed,
        reward_weight=model_stats['Reward Weight']
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
    save_evalutation_stats(env=env, metric=metrics, model_stats=model_stats,
                           filename=f'evaluate/seed_{seed}/{path}/statistics.json')

    # Save the trajectory
    visualize_trajectory(
        initial_positons, episode_actions, f'evaluate/seed_{seed}/{path}/trajectory.mp4')
