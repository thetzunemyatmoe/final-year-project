import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from environment import MultiAgentGridEnv
import json


class ActorNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)  # Softmax for action probabilities
        )

    def forward(self, state):
        action_probs = self.actor(state)
        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single value (state value)
        )

    def forward(self, state):
        value = self.value(state)
        return value

# Define the Agent class that uses separate actor and critic networks


class A3C3Agent:
    def __init__(self, state_size, action_size, learning_rate, beta, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.total_step = 0

        # Actor and Critic networks
        self.actor = ActorNetwork(input_dim, action_dim)
        self.critic = CriticNetwork(input_dim)

        # Optimizers for the actor and critic networks
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate)

    def act(self, state):
        # Select action based on the actor's policy
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantage(self, reward, value_estimate, next_value_estimate, done):
        # Calculate advantage: A3C3 Advantage = R_j - V(O_j^i)
        if done:
            return reward - value_estimate
        else:
            return reward + next_value_estimate - value_estimate

    def update(self, state, action, reward, next_state, done, log_prob):
        # Get value estimate and next value estimate from the critic
        value_estimate = self.critic(state)
        next_value_estimate = self.critic(next_state)

        # Compute advantage using the reward and value estimates
        advantage = self.compute_advantage(
            reward, value_estimate, next_value_estimate, done)

        # Value loss (mean squared error)
        value_loss = F.mse_loss(
            value_estimate, reward + next_value_estimate * (1 - done))

        # Actor loss (Policy Gradient + Entropy Regularization)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        entropy = dist.entropy().mean()

        # Compute actor loss (negative log probability * advantage + entropy term)
        actor_loss = -log_prob * advantage - self.beta * entropy

        # Total loss (sum of value loss and actor loss)
        total_loss = value_loss + actor_loss

        # Backpropagate and optimize the actor and critic
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        # Update the actor and critic networks
        self.actor_optimizer.step()
        self.critic_optimizer.step()


def train(num_episodes, learning_rate, gamma, beta, N, batch_size, step_counter):

    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4,
        initial_positions=[(25, 25), (26, 25), (25, 26), (26, 26)]
    )
    t = 0
    # Sizes to construct the neural network
    state_size = env.get_obs_size()
    action_size = env.get_total_actions()

    state_size, action_size, learning_rate, beta, gamma
    agents = [A3C3Agent(state_size, action_size,
                        learning_rate=0.001, beta=0.001, gamma=0.001)]

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        t_start = t

        while not done or not t - t_start == batch_size:
            actions = [agent.act(state[i], sensor_readings[i])
                       for i, agent in enumerate(agents)]
            next_state, rewards, done, actual_actions = env.step(actions)
            episode_actions.append(actual_actions)
            t += 1

        # #  rewards -> [List of each UAV reward]
        # for i, agent in enumerate(agents):


env = MultiAgentGridEnv(
    grid_file='grid_world.json',
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4,
    initial_positions=[(25, 25), (26, 25), (25, 26), (26, 26)]
)
