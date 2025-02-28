import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from environment import MultiAgentGridEnv


class Actor(nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.actor(state)


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_obs):
        return self.critic(global_obs)


class GlobalActorCritic(nn.Module):
    def __init__(self, actor_input_size, critic_input_size, action_size):
        super(GlobalActorCritic, self).__init__()

        # Single shared actor and critic for all agents
        self.actor = Actor(actor_input_size, action_size)
        self.critic = Critic(critic_input_size)

    def forward(self, global_obs, partial_obs):
        # Critic
        global_obs = [i for obv in global_obs for i in obv]
        global_state = torch.tensor(global_obs, dtype=torch.float32)
        value = self.critic(global_state)  # Shared centralized critic

        # Actor
        partial_state = torch.tensor(partial_obs, dtype=torch.float32)
        policy = self.actor(partial_state)  # Shared actor for all agents

        return policy, value


class Worker(mp.Process):
    def __init__(self, global_model, optimizer, num_agents, num_episode, gamma=0.99, beta=0.01, t_max=5):
        super(Worker, self).__init__()

        # The environment
        self.env = MultiAgentGridEnv(
            grid_file='grid_world_test.json',
            coverage_radius=3,
            max_steps_per_episode=50,
            initial_positions=[(0, 0), (0, 1), (0, 2), (0, 3)]
        )
        # print(f'Environement without UAVs \n {self.env.grid}')
        # print(f'This is in Worker constructer \n {self.env.coverage_grid}')

        # Global model and the oprimizer
        self.global_model = global_model
        self.optimizer = optimizer

        # Variables
        self.num_agents = num_agents
        self.gamma = gamma
        self.beta = beta
        self.t_max = t_max
        self.num_episode = num_episode

    def run(self):
        global_state = self.env.reset()
        # print(f'This is in Worker constructer \n {self.env.coverage_grid}')

        print('-------------------------')

        for _ in range(self.num_episode):

            log_probs = [[] for _ in range(self.num_agents)]
            values = [[] for _ in range(self.num_agents)]
            rewards = [[] for _ in range(self.num_agents)]

            # Collect mini-batch of experiences
            for _ in range(self.t_max):
                actions = []
                # Actions selection
                for agent_id in range(self.num_agents):
                    # Feed the global state and agent index into actor and critic
                    policy, value = self.global_model(
                        global_state, global_state[agent_id])

                    action = torch.argmax(policy, 0).item()
                    # Log probabiltiy for actor loss
                    log_probs[agent_id].append(torch.log(policy[action]))
                    # Value for actor and critic loss
                    values[agent_id].append(value)
                    # Actions chosen
                    actions.append(action)

                next_global_state, reward, done, _ = self.env.step(actions)
                # global_reward.append(sum(reward))

                for agent_id in range(self.num_agents):
                    # Separate reward per agent
                    rewards[agent_id].append(reward[agent_id])

                global_state = next_global_state

            # Compute Advantage and Returns
            returns = [[] for _ in range(self.num_agents)]
            for agent_id in range(self.num_agents):
                #
                R = 0 if done else self.global_model(
                    global_state, global_state[agent_id])[1].item()
                for reward in reversed(rewards[agent_id]):
                    R = reward + self.gamma * R
                    returns[agent_id].insert(0, R)
                returns[agent_id] = torch.tensor(
                    returns[agent_id], dtype=torch.float32)

            # Compute Losses
            actor_losses = []
            critic_losses = []

            for agent_id in range(self.num_agents):
                # Calculating advantage
                advantage = returns[agent_id] - torch.cat(values[agent_id])
                actor_loss = - \
                    (torch.stack(log_probs[agent_id])
                     * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                entropy = - \
                    torch.sum(torch.stack(
                        log_probs[agent_id]) * torch.exp(torch.stack(log_probs[agent_id])))

                actor_losses.append(actor_loss - self.beta * entropy)
                critic_losses.append(critic_loss)

            total_loss = sum(actor_losses) + sum(critic_losses)

            # Update Global Model
            self.optimizer.zero_grad()
            total_loss.backward()
            for global_param, local_param in zip(self.global_model.parameters(), self.global_model.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()

        print(self.env.agent_positions)
        print(self.env.coverage_grid)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    env = MultiAgentGridEnv(
        grid_file='grid_world_test.json',
        coverage_radius=3,
        max_steps_per_episode=50,
        initial_positions=[(0, 1), (0, 2), (0, 3), (0, 4)]
    )

    global_model = GlobalActorCritic(
        actor_input_size=env.get_obs_size(
        ),
        critic_input_size=env.get_obs_size() * env.num_agents, action_size=env.get_total_actions())
    global_model.share_memory()

    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    # ---------------------

    workers = []
    for worker_id in range(1):
        worker = Worker(
            global_model, optimizer, env.num_agents, num_episode=100)
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()
