from environment import MultiAgentGridEnv
from IA2CC import IA2CC
import numpy as np
from utils import display_plot, save_model_stats
import time
from datetime import timedelta
import pandas as pd

GRID_FILE = 'grid_world.json'


def get_new_rollout():
    return [], [], [], [], []


def train(max_episode=5000, actor_lr=0.0001, critic_lr=0.001, gamma=0.999, entropy_weight=0.05, path=None, reward_weight=None):

    # Start training time
    start_time = time.time()

    env = MultiAgentGridEnv(
        grid_file=GRID_FILE,
        coverage_radius=4,
        max_steps_per_episode=150,
        num_agents=4,
        reward_weight=reward_weight
    )

    # NN pararmeters
    critic_input_size = env.get_state_size()
    actor_input_size = env.get_obs_size()
    actor_output_size = env.get_total_actions()

    ia2cc = IA2CC(actor_input_size=actor_input_size,
                  actor_output_size=actor_output_size,
                  critic_input_size=critic_input_size,
                  num_agents=env.num_agents,
                  actor_learning_rate=actor_lr,
                  critic_learning_rate=critic_lr,
                  gamma=gamma,
                  entropy_weight=entropy_weight,
                  )

    episodes_reward = []
    episodes = []

    for episode in range(max_episode):
        joint_observations, state = env.reset(train=episode)
        total_reward = 0
        done = False
        episode_actions = []

        # Rollout
        obs_buffer, next_obs_buffer, log_probs_buffer, entropies_buffer, rewards_buffer = get_new_rollout()

        while not done:

            # Choose action
            sensor_readings = env.get_sensor_readings()

            actions, log_probs, entropies = ia2cc.act(
                joint_observations, sensor_readings)

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

        if episode % 1000 == 0:
            print(f"Episode {episode} return: {total_reward:.2f}")

        episodes.append(episode)
        # Normalize rewards_buffer
        mean_r = np.mean(rewards_buffer)
        std_r = np.std(rewards_buffer) + 1e-8
        normalized_rewards = [(r - mean_r) / std_r for r in rewards_buffer]

        # Value of the terminal state
        last_value = 0

        ia2cc.compute_episode_loss(
            normalized_rewards,
            obs_buffer,
            log_probs_buffer,
            entropies_buffer,
            last_value,
            gamma=ia2cc.gamma,
            entropy_weight=ia2cc.entropy_weight
        )

        # New Rollout
        obs_buffer, next_obs_buffer, log_probs_buffer, entropies_buffer, rewards_buffer = get_new_rollout()

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_taken = str(timedelta(seconds=elapsed_time))

    # Save model and training stats
    if path is not None:
        ia2cc.save_actors(directory=f'model/{path}')
        save_model_stats(name='Independent Advantage Actor Centralized Critic (IA2CC)',
                         model=ia2cc, env=env, max_episode=max_episode, time=time_taken, filename=f"model/{path}/model_stats.json")

    return ia2cc, episodes_reward, episodes


if __name__ == '__main__':

    # Reward weight
    reward_weight = {
        'total area weight': 12.0,
        'overlap weight': 0.8,
        'energy weight': 0.8
    }

    # Record for plot
    rewards_list = []
    episodes_list = []
    names = []

    # Start training
    _, rewards, episodes = train(max_episode=3000, actor_lr=0.0001,
                                 critic_lr=0.001, gamma=0.999, entropy_weight=0.05, path='highreward', reward_weight=reward_weight)

    # Plot
    rewards_list.append(rewards)
    episodes_list.append(episodes)
    names.append('VOID')
    display_plot(rewards_list=rewards_list,
                 episodes_list=episodes_list,
                 names=names,
                 plot_title='Reward Trend',
                 filename='Reward trend in training',
                 save=True)
