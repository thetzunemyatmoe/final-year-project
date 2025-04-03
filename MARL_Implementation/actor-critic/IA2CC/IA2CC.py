import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


from Actor import Actor
from Critic import Critic


class IA2CC:
    def __init__(self, actor_input_size, actor_output_size, critic_input_size, num_agents, actor_learning_rate=0.0001, critic_leanring_rate=0.005, gamma=0.99, entropy_weight=0.05):
        # NN pararmeters
        self.actor_input_size = actor_input_size
        self.actor_output_size = actor_output_size
        self.critic_input_size = critic_input_size
        self.num_agents = num_agents

        print(f'Actor input size is {actor_input_size}')
        print(f'Critic input size is {critic_input_size}')
        # Networks
        self.central_critic = Critic(input_size=self.critic_input_size)
        self.actors = [Actor(self.actor_input_size, self.actor_output_size)
                       for _ in range(self.num_agents)]

        # Optimizer
        self.actor_optimizers = [optim.Adam(
            actor.parameters(), lr=actor_learning_rate) for actor in self.actors]
        self.critic_optimizer = optim.Adam(
            self.central_critic.parameters(), lr=critic_leanring_rate)

    def act(self, joint_observation):
        actions = []
        log_probs = []
        entropies = []
        for agent_id, actor in enumerate(self.actors):
            action, log_prob, entropy = actor(joint_observation[agent_id])
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        return actions, log_probs, entropies

    def get_value(self, state):
        return self.central_critic.forward(state)

    def compute_episode_loss(self, rewards, states, log_probs, entropies, last_value, gamma=0.99, entropy_weight=0.01):
        T = len(rewards)

        # All state values
        values = [self.get_value(state) for state in states]
        last_value = torch.tensor(last_value)
        values.append(last_value.detach())

        # Advantages
        advantages = []
        for t in range(T):
            delta = rewards[t] + gamma * values[t+1] - values[t]
            advantages.append(delta)

        # Critic loss
        critic_loss = sum([adv.pow(2) for adv in advantages]) / T
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        for i in range(self.num_agents):
            actor_loss = 0
            for t in range(T):
                advantage = advantages[t].detach()
                entropy = entropies[t][i]  # entropy for agent i at timestep t
                actor_loss += -log_probs[t][i] * \
                    advantage - entropy_weight * entropy
            actor_loss /= T
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

    def moving_average(self, x, count):
        return np.convolve(x, np.ones((count,)) / count, mode='valid')

    def display_moving_average(self, episodes_reward):
        plt.plot(self.moving_average(episodes_reward, 100))
        plt.title('Learning curve')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.show()
