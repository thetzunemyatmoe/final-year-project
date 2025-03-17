from simulator.GymNavigation import GymNav
from GlobalActorCritic import GlobalActorCritic
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp


class Worker(mp.Process):
    def __init__(self, global_model, actor_optimizer, critic_optimizer, num_agents, max_num_episode, gamma=0.99, beta=0.01, batch_size=25):
        super(Worker, self).__init__()

        self.env = GymNav()

        # Local model
        self.local_model = GlobalActorCritic(
            actor_input_size=env.agent_observation_space,
            critic_input_size=env.central_observation_space,
            action_size=env.agent_action_space
        )

        # Global model
        self.global_model = global_model

        # Optimizer for actor and critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Variables
        self.num_agents = num_agents
        self.gamma = gamma
        self.beta = beta
        self.batch_size = batch_size
        self.max_num_episode = max_num_episode
        self.number_of_agents = env.number_of_agents

    # Syncing the parameter of local model with global model's parameter
    def sync_local_with_global(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    def run(self):
        self.sync_local_with_global()

        episode_buffer = [[] for _ in range(self.number_of_agents)]
        episode_values = [[] for _ in range(self.number_of_agents)]
        episode_reward = 0
        episode_step_count = 0

        agent_states, global_states = self.env.reset()

        for episode_step_count in range(self.max_num_episode):

            # Actor run
            actions = []
            for index, agent_state in enumerate(agent_states):
                with torch.no_grad():
                    action_distribution = self.local_model.get_policy(
                        agent_state)
                action_distribution = action_distribution.squeeze(
                    0)  # Remove extra dimensions if needed

                # Sample one action based on the probability distribution
                action = torch.multinomial(
                    action_distribution, num_samples=1).item()

                actions.append(action)

            # Critic run
            values = []
            for index, global_state in enumerate(global_states):
                with torch.no_grad():
                    value = self.local_model.get_value(global_state)

                values.append(value)

            agent_states_before_step = agent_states
            global_states_before_step = global_states

            agent_states, reward, terminal, global_states = self.env.step(
                actions)

            episode_reward += reward
            for agent_index in range(self.num_agents):
                episode_buffer[agent_index].append([
                    agent_states_before_step[agent_index],
                    global_states_before_step[agent_index],
                    reward,
                    agent_states[agent_index],
                    terminal,
                    values[index]

                ])
            if terminal:
                print(episode_step_count)
                break

            if len(episode_buffer[0]) == self.batch_size and not terminal and \
                    episode_step_count < self.max_num_episode - 1:
                for step in episode_buffer[0]:
                    print(step)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    env = GymNav()

    global_model = GlobalActorCritic(
        actor_input_size=env.agent_observation_space,
        critic_input_size=env.central_observation_space,
        action_size=env.agent_action_space
    )
    global_model.share_memory()

    actor_optimizer = optim.Adam(global_model.get_actor_params(), lr=0.0001)
    critic_optimizer = optim.Adam(global_model.get_critic_params(), lr=0.0001)

    workers = []
    for worker_id in range(1):
        worker = Worker(
            global_model, actor_optimizer, critic_optimizer, env.number_of_agents, max_num_episode=10000000)
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()
