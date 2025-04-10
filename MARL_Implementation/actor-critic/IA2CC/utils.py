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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def save_best_episode(initial_positions, best_episode_actions, best_episode_number, best_episode_reward, filename='best_strategy.json'):
    action_map = ['forward', 'backward', 'left', 'right', 'stay']

    best_episode = {
        # Convert to int if it's np.int64
        "episode_number": int(best_episode_number),
        # Convert to float if it's np.float64
        "episode_reward": float(best_episode_reward)

    }

    for i in range(len(initial_positions)):
        best_episode[f'agent_{i}'] = {
            'actions': [action_map[action[i]] for action in best_episode_actions],
            'initial_position': initial_positions[i]
        }

    with open(filename, 'w') as f:
        json.dump(best_episode, f, indent=4)

    print(f"Best episode actions and initial positions saved to {filename}")


def save_final_positions(env, best_episode_actions, filename='final_positions.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()

    for actions in best_episode_actions:
        env.step(actions)

    env.render(
        ax, actions=best_episode_actions[-1], step=len(best_episode_actions)-1)
    plt.title("Final Positions")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final positions saved as {filename}")


def visualize_trajectory(env, episode_actions, filename=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()

    if filename is not None:
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
        print(f"Best episode visualization saved as {filename}")
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


def visualize_best_trajectory(initial_positions, episode_actions, filename=None):
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
        print(f"Best episode visualization saved as {filename}")
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


def evaluate(model, episode_count=50):

    episode_reward = []
    episode_energy = []

    for episode in range(50, episode_count+50):
        reward, energy = run_episode(episode, model)
        episode_reward.append(reward)
        episode_energy.append(energy)

    print(
        f"Average reward over environments: {np.mean(episode_reward):.2f} ± {np.std(episode_reward):.2f}")
    print(
        f"Average energy usage over environments: {np.mean(episode_energy):.2f} ± {np.std(episode_energy):.2f}")


def run_episode(seed, model):
    env = MultiAgentGridEnv(
        grid_file=GRID_FILE,
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4,
        seed=seed
    )

    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        sensor_readings = env.get_sensor_readings()
        actions, _, _ = model.act(obs, sensor_readings)
        obs, r, done, _, _ = env.step(actions)
        total_reward += r

    return total_reward, env.get_metrics()["Energy Usage"]
