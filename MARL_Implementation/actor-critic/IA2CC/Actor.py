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

    def forward(self, local_observation, sensor_readings):
        # Include 'stay' action
        mask = sensor_readings + [0]
        logits = self.get_logit(local_observation)
        action_mask = torch.tensor(
            mask, dtype=torch.uint8, device=logits.device)

        # 0 = valid → True, 1 = invalid → False
        valid_mask = (action_mask == 0).to(
            dtype=torch.bool, device=logits.device)

        logits = logits.masked_fill(~valid_mask, float('-inf'))

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, dist.entropy()

    def build_input(self, local_observation):
        return torch.from_numpy(local_observation).float()

    def get_logit(self, local_observation):
        return self.network((self.build_input(local_observation)))

    def extract_model_info(model):
        layers = []
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                layers.append({
                    "type": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features
                })
            elif isinstance(layer, nn.ReLU):
                layers.append({"type": "ReLU"})
            else:
                layers.append({"type": layer.__class__.__name__})
        return layers
