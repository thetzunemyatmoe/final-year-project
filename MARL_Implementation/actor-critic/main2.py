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
                 global_model, actor_optimizer, critic_optimizer, num_agents, num_episode, grid_file, coverage_radius, max_steps_per_episode, intial_positions,
                 gamma=0.99, beta=0.01, t_max=5):
        super(Worker, self).__init__()

        # The environment
        self.env = MultiAgentGridEnv(
            grid_file=grid_file,
            coverage_radius=coverage_radius,
            max_steps_per_episode=max_steps_per_episode,
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

            # Collect mini-batch of experiences
            for t in range(self.t_max):
                actions = [0] * self.num_agents
                # Actions selection (From actor)
                for agent_id in range(self.num_agents):
                    # Feed the global state and partial observation the local model
                    policy, value, policy_entropy = self.local_model(
                        global_state, global_state[agent_id])

                    action = torch.argmax(policy, 0).item()
                    # Log probabiltiy and Entropy for actor loss in timestep t
                    log_probs[agent_id][t] = torch.log(policy[action])
                    policy_entropies[agent_id][t] = policy_entropy

                    # Value for critic loss
                    values[agent_id][t] = value
                    # Actions chosen
                    actions[agent_id] = action

                # Execurint actions
                next_global_state, reward, done, _ = self.env.step(
                    actions, t, episode)

                # Storing reward for each agent in each timestep
                for agent_id in range(self.num_agents):
                    # Separate reward per agent
                    rewards[agent_id][t] = reward[agent_id]

                global_state = next_global_state

            # Compute Advantage and Returns(Value)
            returns = [[0]*self.t_max for _ in range(self.num_agents)]
            for agent_id in range(self.num_agents):
                # Value of the current state
                R = 0 if done else self.local_model(
                    global_state, global_state[agent_id])[1].item()

                # Calculating value of
                for time, reward in reversed(list(enumerate(rewards[agent_id]))):

                    R = reward + self.gamma * R
                    returns[agent_id][time] = R

                returns[agent_id] = torch.tensor(
                    returns[agent_id], dtype=torch.float32)

            # Compute Losses
            actor_losses = [[0]*self.t_max for _ in range(self.num_agents)]
            critic_losses = [[0]*self.t_max for _ in range(self.num_agents)]

            for agent_id in range(self.num_agents):
                # For each step backward
                for time in range(self.t_max-1, -1, -1):

                    advantage = returns[agent_id][time] - \
                        values[agent_id][time]

                    # Calculating critic (value) loss
                    critic_loss = advantage.pow(2)
                    critic_losses[agent_id][time] = critic_loss

                    # Calculating actor loss
                    actor_loss = log_probs[agent_id][time] * \
                        advantage - policy_entropies[agent_id][time]
                    actor_losses[agent_id][time] = actor_loss

            # Resetting gradient
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Loop over agents
            for agent_id in range(self.num_agents):
                for t in reversed(range(self.t_max)):  # Backward through time
                    # Compute actor and critic loss for this agent at timestep t
                    actor_loss = actor_losses[agent_id][t]
                    critic_loss = critic_losses[agent_id][t]

                    # Compute gradients separately
                    critic_loss.backward(retain_graph=True)
                    actor_loss.backward()

                    # Accumulate gradients across all timesteps
                    for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                        if local_param.grad is not None:
                            if global_param.grad is None:
                                global_param.grad = local_param.grad.clone()
                            else:
                                global_param.grad += local_param.grad.clone()
            # Apply updates to global model
            self.critic_optimizer.step()
            self.actor_optimizer.step()

        print(self.env.agent_positions)
        print(self.env.coverage_grid)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    grid_file = 'grid_world_test.json'
    coverage_radius = 1
    max_steps_per_episode = 50
    initial_positions = [(0, 0), (0, 1)]

    env = MultiAgentGridEnv(
        grid_file=grid_file,
        coverage_radius=coverage_radius,
        max_steps_per_episode=max_steps_per_episode,
        initial_positions=initial_positions
    )

    global_model = GlobalActorCritic(
        actor_input_size=env.get_obs_size(),
        critic_input_size=env.get_obs_size() * env.num_agents,
        action_size=env.get_total_actions())
    global_model.share_memory()

    actor_optimizer = optim.Adam(global_model.get_actor_params(), lr=0.0005)
    critic_optimizer = optim.Adam(global_model.get_critic_params(), lr=0.001)

    workers = []
    for worker_id in range(1):
        worker = Worker(
            global_model, actor_optimizer, critic_optimizer, env.num_agents, 10, grid_file, coverage_radius, max_steps_per_episode, initial_positions)
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()
