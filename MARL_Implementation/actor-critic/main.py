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
    def __init__(self, actor_state_size, action_size, critic_state_size, learning_rate, beta, gamma):
        self.actor_state_size = actor_state_size
        self.critic_state_size = critic_state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.total_step = 0

        # Actor and Critic networks
        self.actor = ActorNetwork(self.actor_state_size, self.action_size)
        self.critic = CriticNetwork(self.critic_state_size)

        # Optimizers for the actor and critic networks
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate)

    def actor_act(self, state):
        state = torch.tensor(state)
        # Select action based on the actor's policy
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def critic_act(self, state):
        state = [i for obv in state for i in obv]
        state = torch.tensor(state)

        # Extrating value from critic network
        value = self.critic(state)

        return value

    def compute_advantage(self, reward, value_estimate, next_value_estimate, done):
        # Calculate advantage: A3C3 Advantage = R_j - V(O_j^i)
        return reward + next_value_estimate - value_estimate

    def compute_loss(self, global_state_value, global_state):

        # Value loss
        global_state_value = torch.tensor(global_state_value)
        global_state = [i for obv in global_state for i in obv]
        global_state = torch.tensor(global_state)
        value = self.critic(global_state)
        value_loss = (global_state_value - value).pow(2).mean()

        return value_loss, actor_loss

    def update(self, global_state_value, memory, index):
        reward = memory['rewards'][index]
        global_state_value = reward + global_state_value

        value_loss, actor_loss = self.compute_loss(
            global_state_value, memory["global_state"])


def train(num_episodes, learning_rate, gamma, beta, N, batch_size, step_counter):

    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=4,
        max_steps_per_episode=50,
        initial_positions=[(25, 25), (26, 25), (25, 26), (26, 26)]
    )
    t = 0
    # Sizes to construct the neural network
    actor_state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    critic_state_size = actor_state_size * \
        env.num_agents  # Currently unpack the array

    agents = [A3C3Agent(actor_state_size, action_size, critic_state_size,
                        learning_rate=0.001, beta=0.001, gamma=0.001)]

    for episode in range(num_episodes):
        global_state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        t_start = t

        memory = {}
        while not done or not t - t_start == batch_size:
            actions = [agent.actor_act(global_state[i])  # sensor_readings[i]) is sensor necessary
                       for i, agent in enumerate(agents)]
            next_gloabl_state, rewards, done, actual_actions = env.step(
                actions)
            episode_actions.append(actual_actions)
            t += 1

            memory[t]["global_state"] = global_state
            memory[t]["next_global_state"] = next_gloabl_state
            memory[t]["rewards"] = rewards

            global_state = next_gloabl_state

        # #  rewards -> [List of each UAV reward]
        for index, agent in enumerate(agents):
            global_state_value = 0 if done else agent.critic_act(global_state)

            # Backward
            for time in range(t-1, t_start, -1):
                agents.update(global_state_value, memory[time], index)
