from environment import MultiAgentGridEnv
from IA2CC import IA2CC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import save_best_episode, save_final_positions, visualize_and_record_best_strategy


GRID_FILE = 'grid_world.json'


def evaluate(ia2cc, environment_count=1000, envs=None):

    # If environment are not provides
    if envs is None:
        envs = [MultiAgentGridEnv(
            grid_file=GRID_FILE,
            coverage_radius=4,
            max_steps_per_episode=50,
            num_agents=4
        ) for _ in range(environment_count)]
    else:
        environment_count = len(envs)

    rewards = []

    for env in envs:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            actions, _, _ = ia2cc.act(obs)
            obs, r, done, _, _ = env.step(actions)
            total_reward += r
        rewards.append(total_reward)

    print(
        f"\n✅ Avg evaluation reward over {environment_count} environments: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


def get_new_rollout():
    return [], [], [], [], []


def train(max_episode=3000, actor_lr=1e-4, critic_lr=5e-3, gamma=0.99, entropy_weight=0.05):

    env = MultiAgentGridEnv(
        grid_file=GRID_FILE,
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4,
        # initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
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
                  critic_leanring_rate=critic_lr,
                  gamma=gamma,
                  entropy_weight=entropy_weight
                  )

    episodes_reward = []
    episodes = []
    best_episode_reward = float('-inf')
    best_episode_actions = None
    best_episode_number = None

    for episode in range(max_episode):
        joint_observations, state = env.reset(train=True)
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

        if episode % 1000 == 0:
            print(f"Episode {episode} return: {total_reward:.2f}")

        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            best_episode_actions = episode_actions
            best_episode_number = episode

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
            entropy_weight=ia2cc.entropy_weight
        )

        # New Rollout
        obs_buffer, next_obs_buffer, log_probs_buffer, entropies_buffer, rewards_buffer = get_new_rollout()

    # evaluate(ia2cc=ia2cc)
    # save_best_episode(env.initial_positions, best_episode_actions,
    #                   best_episode_number, best_episode_reward)
    # save_final_positions(env, best_episode_actions)
    # visualize_and_record_best_strategy(env, best_episode_actions)

    return ia2cc, episodes_reward, episodes


if __name__ == '__main__':
    train()
