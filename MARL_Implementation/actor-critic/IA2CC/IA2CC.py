import torch.optim as optim
from Actor import Actor
from Critic import Critic


class IA2CC:
    def __init__(self, actor_input_size, actor_output_size, critic_input_size, num_agents, actor_learning_rate=0.001, critc_leanring_rate=0.005):
        # NN pararmeters
        self.actor_input_size = actor_input_size
        self.actor_output_size = actor_output_size
        self.critic_input_size = critic_input_size

        # Networks
        self.central_critic = Critic(input_size=self.critic_input_size)
        self.actors = [Actor(self.actor_input_size, self.actor_output_size)
                       for _ in range(num_agents)]

        # Optimizer
        self.actor_optimizers = [optim.Adam(
            actor.parameters(), lr=actor_learning_rate) for actor in self.actors]
        self.critic_optimizer = optim.Adam(
            self.central_critic.parameters(), lr=critc_leanring_rate)

    def act(self, joint_observation):
        actions = []
        for agent_id, actor in enumerate(self.actors):
            action = actor.forward(joint_observation[agent_id])
            actions.append(action)

        return actions
