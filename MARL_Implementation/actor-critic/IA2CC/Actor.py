import torch
import torch.nn as nn
from torch.distributions import Categorical


# Input : Local observation (obs_size)
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, local_observation):
        logits = self.get_logit(local_observation)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, dist.entropy()

    def build_input(self, local_observation):
        return torch.from_numpy(local_observation).float()

    def get_logit(self, local_observation):
        return self.network((self.build_input(local_observation)))
