import torch
import torch.optim as optim
import torch.nn.functional as F

from Actor import Actor
from Critic import Critic
from Memory import Memory


class IA2CC:
    def __init__(self, actor_input_size, actor_output_size, critic_input_size, num_agents, actor_learning_rate=0.001, critc_leanring_rate=0.005):
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
            self.central_critic.parameters(), lr=critc_leanring_rate)

        # Memory
        self.memory = Memory(agent_num=num_agents,
                             action_dim=actor_output_size)

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

    def get_value(self, joint_observation):
        return self.central_critic.forward(joint_observation=joint_observation)

    def compute_loss(self, reward, joint_observations, next_joint_observations, log_probs, entropies, entropy_weight=0.1):
        # --- Critic update ---
        current_value = self.get_value(joint_observation=joint_observations)
        next_value = self.get_value(joint_observation=next_joint_observations)

        if not torch.is_tensor(reward):
            reward = torch.tensor(
                [[reward]], dtype=torch.float32, device=current_value.device)

        td_error = reward + 0.99 * next_value - current_value
        critic_loss = td_error.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # --- Actor update with entropy regularization ---
        for i in range(self.num_agents):
            advantage = td_error.detach().view(-1)[0]  # Safe scalar
            actor_loss = -log_probs[i] * advantage
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        return critic_loss
