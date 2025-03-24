import torch
import torch.nn as nn


# Input : Joint Observation (n * obs_size)
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, joint_observation):
        x = self.build_state(joint_observation)

        return self.network(x)

    def build_state(self, joint_observation):
        obs_tensors = [torch.tensor(obs, dtype=torch.float32)
                       for obs in joint_observation]
        state_tensor = torch.cat(obs_tensors, dim=0)
        return state_tensor.unsqueeze(0)
