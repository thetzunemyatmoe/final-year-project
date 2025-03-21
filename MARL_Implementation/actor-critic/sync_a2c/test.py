import multiprocessing as mp
from global_env import MultiAgentGridEnv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Networks import Actor, Critic


gamma = 0.99
num_envs = 5  # Number of parallel environments
max_episode = 100  # Number of timesteps per rollout

initial_positions = [(0, 0), (0, 7), (3, 3), (7, 0), (7, 7)]
# Initialize K parallel environments
envs = [MultiAgentGridEnv(grid_file="grid_world.json", coverage_radius=1, max_steps_per_episode=200,
                          num_agents=1, initial_positions=[initial_positions[index]]) for index in range(num_envs)]

actor = Actor(envs[0].get_obs_size(), envs[0].get_total_actions())
critic = Critic(envs[0].get_obs_size())
actor_optim = optim.Adam(actor.parameters(), lr=0.0005)
critic_optim = optim.Adam(critic.parameters(), lr=0.001)


def env_reset(envs):
    states = []
    for env in envs:
        state = env.reset()
        states.append(state)

    return states


def sample_action(envs, states):

    actions = []
    for index, env in enumerate(envs):
        action = []
        for agent in range(env.num_agents):
            # Action
            action_probs = actor.forward(torch.Tensor(states[index][agent]))
            dist = Categorical(action_probs)
            action.append(dist.sample().item())

        actions.append(action)

    return actions


def all_step(envs, actions):
    next_states = []
    rewards = []
    for index, env in enumerate(envs):
        states, reward, done, actual_action = env.step(actions[index])

        next_states.append(states)
        rewards.append(reward)

    return next_states, rewards, done


def calculate_loss(states, actions, rewards, done, next_states, beta=0.5):
    actor_loss, critic_loss, entropy_loss = 0, 0, 0

    for index, env in enumerate(envs):
        for agent_index in range(env.num_agents):
            state = torch.tensor(
                states[index][agent_index], dtype=torch.float32)
            next_state = torch.tensor(
                next_states[index][agent_index], dtype=torch.float32)
            reward = torch.tensor(
                rewards[index], dtype=torch.float32)  # shape [1]
            done_tensor = torch.tensor([done], dtype=torch.float32)

            # Values
            value = critic(state)  # shape [1]
            next_value = critic(next_state).detach()  # shape [1]

            # TD Target and Advantage
            td_target = reward + gamma * next_value * (1 - done_tensor)
            advantage = td_target - value

            # Critic loss
            critic_loss += F.mse_loss(value, td_target)

            # Actor loss
            action_probs = actor(state)
            dist = Categorical(action_probs)
            selected_action = torch.tensor(
                actions[index][agent_index], dtype=torch.int64)
            log_prob = dist.log_prob(selected_action)
            entropy = dist.entropy()

            actor_loss += -log_prob * advantage.detach()
            entropy_loss += entropy

    total_actor_loss = actor_loss - beta * entropy_loss

    return total_actor_loss, critic_loss

    # Repeat for every episode
for episode in range(max_episode):

    # Observe a batch of current states for all environments
    states = env_reset(envs)

    done = False
    episodic_rewards = [0.0 for _ in range(num_envs)]

    # for time step t = 0, 1, 2, . . . do
    while not done:

        # Sample actions
        actions = sample_action(envs, states)

        # Step
        next_states, rewards, done = all_step(envs, actions)

        actor_loss, critic_loss = calculate_loss(
            states, actions, rewards, done, next_states)

        # Accumulate episodic rewards
        for j in range(num_envs):
            episodic_rewards[j] += rewards[j]

        # Actor update
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # Critic update
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        states = next_states

    # Print episodic results
    print(envs[2].poi_coverage_counter)
    avg_reward = sum(episodic_rewards) / num_envs
    print(f"Episode {episode} finished. Avg Reward: {avg_reward:.2f}")
