import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from environment import MultiAgentGridEnv
import matplotlib.pyplot as plt
import math


class Actor(nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 20),  # First hidden layer (20 neurons)
            nn.ReLU(),
            nn.Linear(20, 10),         # Second hidden layer (10 neurons)
            nn.ReLU(),
            nn.Linear(10, action_size),  # Output layer (action_size neurons)
            nn.Softmax(dim=-1)          # Probability distribution over actions
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Glorot Initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Set bias to zero

    def forward(self, state):
        return self.actor(state)


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, 20),  # First hidden layer (20 neurons)
            nn.ReLU(),
            nn.Linear(20, 10),         # Second hidden layer (10 neurons)
            nn.ReLU(),
            # Output layer (1 neuron for state value)
            nn.Linear(10, 1)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Glorot Initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Set bias to zero

    def forward(self, global_obs):
        return self.critic(global_obs)


class GlobalActorCritic(nn.Module):
    def __init__(self, actor_input_size, critic_input_size, action_size):
        super(GlobalActorCritic, self).__init__()

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

        return policy, value, -self.compute_entropy(policy)

    def compute_entropy(self, action_probs):
        log_probs = torch.log(action_probs + 1e-8)
        entropy = -torch.sum(action_probs * log_probs, dim=-1)

        return entropy

    def get_actor_params(self):
        return self.actor.parameters()

    def get_critic_params(self):
        return self.critic.parameters()


class Worker(mp.Process):
    def __init__(self,
                 global_model, actor_optimizer, critic_optimizer, num_agents, num_episode, grid_file, coverage_radius, intial_positions,
                 gamma=0.99, beta=0.01, t_max=30):
        super(Worker, self).__init__()

        # The environment
        self.env = MultiAgentGridEnv(
            grid_file=grid_file,
            coverage_radius=coverage_radius,
            initial_positions=intial_positions
        )

        # Local model
        self.local_model = GlobalActorCritic(
            actor_input_size=self.env.get_obs_size(),
            critic_input_size=self.env.get_obs_size() * env.num_agents,
            action_size=self.env.get_total_actions())

        # Global model
        self.global_model = global_model

        # Optimizer for actor and critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Variables
        self.num_agents = num_agents
        self.gamma = gamma
        self.beta = beta
        self.t_max = t_max
        self.num_episode = num_episode

    # Syncing the parameter of local model with global model's parameter
    def sync_local_with_global(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    def run(self):
        self.episode_rewards = []

        for episode in range(self.num_episode):
            # Resetting the environment
            global_state = self.env.reset()

            # Synchronzie local model with global model
            self.sync_local_with_global()

            # History storing for loss calculation
            log_probs = [[0]*self.t_max for _ in range(self.num_agents)]
            policy_entropies = [[0]*self.t_max for _ in range(self.num_agents)]
            values = [[0]*self.t_max for _ in range(self.num_agents)]
            rewards = [[0]*self.t_max for _ in range(self.num_agents)]

            episode_reward = 0

            ################
            # Collecting Mini batch
            ################
            for t in range(self.t_max):
                actions = [0] * self.num_agents
                # Actions selection (From actor)
                for agent_id in range(self.num_agents):
                    # Feed the global state and partial observation the local model
                    policy, value, policy_entropy = self.local_model(
                        global_state, global_state[agent_id])

                    action = torch.multinomial(policy, 1).item()
                    # Log probabiltiy and Entropy for actor loss in timestep t
                    log_probs[agent_id][t] = torch.log(policy[action])
                    policy_entropies[agent_id][t] = policy_entropy

                    # Value for critic loss
                    values[agent_id][t] = value
                    # Actions chosen
                    actions[agent_id] = action

                # Execurint actions
                next_global_state, reward, _ = self.env.step(
                    actions, t, episode)

                # Storing reward for each agent in each timestep
                for agent_id in range(self.num_agents):
                    # Separate reward per agent
                    rewards[agent_id][t] = reward[agent_id]
                    # Add reward for this step
                episode_reward += math.ceil(sum(reward) / self.num_agents)

                global_state = next_global_state
            # Store total reward for this episode
            self.episode_rewards.append(episode_reward / self.t_max)

            ################
            # Compute state of the values in each step
            ################
            returns = [[0]*self.t_max for _ in range(self.num_agents)]
            for agent_id in range(self.num_agents):
                # Value of the current state
                R = self.local_model(global_state, global_state[agent_id])[
                    1].item()

                # Calculating value of each state
                for time, reward in reversed(list(enumerate(rewards[agent_id]))):

                    R = reward + self.gamma * R
                    returns[agent_id][time] = R

                returns[agent_id] = torch.tensor(
                    returns[agent_id], dtype=torch.float32)

            ################
            # Loss calculation
            ################

            total_loss = 0.0
            for agent_id in range(self.num_agents):
                for t in range(self.t_max):
                    advantage = returns[agent_id][t] - values[agent_id][t]
                    actor_loss = - \
                        log_probs[agent_id][t] * advantage - \
                        self.beta * policy_entropies[agent_id][t]
                    critic_loss = advantage.pow(2)
                    total_loss += actor_loss + critic_loss

            ################
            # Accumulating gradient
            ################
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Single backward pass over the total loss
            total_loss.backward()

            # Accumulate gradients from the local model to the global model
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                if local_param.grad is not None:
                    if global_param.grad is None:
                        global_param.grad = local_param.grad.clone()
                    else:
                        global_param.grad += local_param.grad.clone()

            ################
            # Update the gloabl model
            ################
            self.critic_optimizer.step()
            self.actor_optimizer.step()

        # print(self.env.agent_positions)
        # print(self.env.coverage_grid)
        # Plot rewards after training
        self.plot_rewards()

    def plot_rewards(self):
        # print(f'The total episode reward is {self.episode_rewards}')
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Total Reward per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Training Reward Trend")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    grid_file = 'grid_world_test.json'
    coverage_radius = 1
    initial_positions = [(0, 1), (4, 3)]  # Two agents

    env = MultiAgentGridEnv(
        grid_file=grid_file,
        coverage_radius=coverage_radius,
        initial_positions=initial_positions
    )

    global_model = GlobalActorCritic(
        actor_input_size=env.get_obs_size(),
        critic_input_size=env.get_obs_size() * env.num_agents,
        action_size=env.get_total_actions())
    global_model.share_memory()

    actor_optimizer = optim.Adam(global_model.get_actor_params(), lr=0.0001)
    critic_optimizer = optim.Adam(global_model.get_critic_params(), lr=0.0001)

    workers = []
    for worker_id in range(mp.cpu_count()):
        worker = Worker(
            global_model, actor_optimizer, critic_optimizer, env.num_agents, 1000, grid_file, coverage_radius, initial_positions)
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()
