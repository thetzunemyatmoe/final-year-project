import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Networks import Actor, Critic


class GlobalActorCritic(nn.Module):
    def __init__(self, actor_input_size, critic_input_size, action_size):
        super(GlobalActorCritic, self).__init__()

        self.actor = Actor(actor_input_size, action_size)
        self.critic = Critic(critic_input_size)

    def get_policy(self, partial_obs):
        partial_state = torch.tensor(partial_obs, dtype=torch.float32)
        policy = self.actor(partial_state)  # Shared actor for all agents

        return policy

    def get_value(self, global_obs):
        global_state = torch.tensor(global_obs, dtype=torch.float32)
        value = self.critic(global_state)  # Shared centralized critic

        return value

    def compute_entropy(self, action_probs):
        log_probs = torch.log(action_probs + 1e-8)
        entropy = -torch.sum(action_probs * log_probs, dim=-1)

        return entropy

    def get_actor_params(self):
        return self.actor.parameters()

    def get_critic_params(self):
        return self.critic.parameters()
