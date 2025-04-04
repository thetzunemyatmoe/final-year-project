from environment import MultiAgentGridEnv
from IA2CC import IA2CC
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import math
import json

grid_file = 'grid_world.json'


def evaluate(ia2cc, episodes=20):
    # Create 20 independent environments
    envs = [MultiAgentGridEnv(
        grid_file=grid_file,
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4
    ) for _ in range(episodes)]

    rewards = []

    for env in envs:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # deterministic=True → greedy
            actions, _, _ = ia2cc.act(obs)
            obs, r, done, _, _ = env.step(actions)
            total_reward += r
        rewards.append(total_reward)

    print(
        f"\n✅ Avg evaluation reward over {episodes} environments: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


def get_new_rollout():
    return [], [], [], [], []


def train(max_episode=3000, actor_lr=1e-4, critic_lr=5e-3, gamma=0.99, entropy_weight=0.05):

    env = MultiAgentGridEnv(
        grid_file=grid_file,
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )

    # NN pararmeters
    critic_input_size = env.get_state_size()
    actor_input_size = env.get_obs_size()
    actor_output_size = env.get_total_actions()

    ia2cc = IA2CC(actor_input_size=actor_input_size,
                  actor_output_size=actor_output_size,
                  critic_input_size=critic_input_size,
                  num_agents=env.num_agents,
                  #   actor_learning_rate=actor_lr,
                  #   critic_leanring_rate=critic_lr,
                  #   gamma=gamma,
                  #   entropy_weight=entropy_weight
                  )

    episodes_reward = []
    best_episode_reward = float('-inf')
    best_episode_actions = None
    best_episode_number = None

    for episode in range(max_episode):
        joint_observations, state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        # Rollout
        obs_buffer, next_obs_buffer, log_probs_buffer, entropies_buffer, rewards_buffer = get_new_rollout()

        while not done:

            # Choose action
            actions, log_probs, entropies = ia2cc.act(joint_observations)

            # Take step
            next_joint_observations, reward, done, actual_actions, state = env.step(
                actions)
            episode_actions.append(actual_actions)

            # Store in buffer
            obs_buffer.append(state)
            next_obs_buffer.append(next_joint_observations)
            log_probs_buffer.append(log_probs)
            entropies_buffer.append(entropies)
            rewards_buffer.append(reward)

            total_reward += reward
            joint_observations = next_joint_observations

        episodes_reward.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode} return: {total_reward:.2f}")
            print(env.agent_positions)

        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            best_episode_actions = episode_actions
            best_episode_number = episode

        # Normalize rewards_buffer
        mean_r = np.mean(rewards_buffer)
        std_r = np.std(rewards_buffer) + 1e-8
        normalized_rewards = [(r - mean_r) / std_r for r in rewards_buffer]

        # Compute value of final state (for bootstrap)
        last_value = 0

        ia2cc.compute_episode_loss(
            normalized_rewards,
            obs_buffer,
            log_probs_buffer,
            entropies_buffer,
            last_value,
        )

        # New Rollout
        obs_buffer, next_obs_buffer, log_probs_buffer, entropies_buffer, rewards_buffer = get_new_rollout()

    evaluate(ia2cc=ia2cc)
    ia2cc.display_moving_average(episodes_reward)
    save_best_episode(env.initial_positions, best_episode_actions,
                      best_episode_number, best_episode_reward)
    save_final_positions(env, best_episode_actions)
    visualize_and_record_best_strategy(env, best_episode_actions)


def save_best_episode(initial_positions, best_episode_actions, best_episode_number, best_episode_reward, filename='vdn_best_strategy.json'):
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


def save_final_positions(env, best_episode_actions, filename='vdn_final_positions.png'):
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


def visualize_and_record_best_strategy(env, best_episode_actions, filename='vdn_best_episode.mp4'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()

    # Set up the video writer
    writer = FFMpegWriter(fps=2)

    with writer.saving(fig, filename, dpi=100):
        # Capture the initial state
        ax.clear()
        env.render(ax, actions=None, step=0)
        writer.grab_frame()
        plt.pause(0.1)

        for step, actions in enumerate(best_episode_actions, start=1):
            env.step(actions)
            ax.clear()
            env.render(ax, actions=actions, step=step)
            writer.grab_frame()
            plt.pause(0.1)

    plt.close(fig)
    print(f"Best episode visualization saved as {filename}")


train()
