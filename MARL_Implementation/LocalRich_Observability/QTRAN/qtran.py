# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import math

from environment import MultiAgentGridEnv
import json

# Sustom neural network class that inherits from PyTorch's nn.Module.


class QTRANQNetwork(nn.Module):
    '''
    input_size -> The number of features in the input data
    output_size -> The number of output

    '''

    def __init__(self, input_size, output_size):
        # Calls the constructor of the parent class nn.Module class to initialize its properties and behavious defined in the base class
        super(QTRANQNetwork, self).__init__()
        # self.network -> the architecture of the neural network as a sequential stack of layers in a linear ordered.
        self.network = nn.Sequential(
            # A fully connected layer with input_size neurons cnnected to 128 neorons in the first hidden layer
            nn.Linear(input_size, 128),
            # Rectified Linear Unit (ReLU) activation function, applied element-wise
            nn.ReLU(),
            # A second fully connected layer with 128 input neurons connected to 128 output neurons
            nn.Linear(128, 128),
            # Rectified Linear Unit (ReLU) activation function, applied element-wise
            nn.ReLU(),
            # The final fully connected layer connects 128 neurons to output_size neurons.
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        '''
        - Defines the forward pass, where the input x is propagated through the network.
        - The forward method is automatically called when the model instance is used to make predictions
        - Feeds the input tensor x through the sequentially defined layers in self.network
        '''
        return self.network - (x)


class QTRANMixer(nn.Module):
    # Initialized two neutal networks
    def __init__(self, state_dim, n_agents, embed_dim=32):
        '''
        state_dim -> dimensionality of the state space
        n_agents -> representing the number oif agents in the system
        embed_dim -> dimensionmality of intermediate feature
        '''
        super(QTRANMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # A neural network to compute the joint Q-value

        self.Q = nn.Sequential(
            nn.Linear(state_dim + n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # A neural network to compute the joint V-value
        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    # Process the inputs (states and agent_qs) to compute the joint Q-value and V-value
    def forward(self, states, agent_qs):
        '''
        states - A tensor to represent the global state of the environment
        action - A tensor to represent the individual Q-values predicted by each agent's policy 
        '''

        # Batch size of the input data;
        bs = states.size(0)

        # Reshape states if necessary
        if len(states.shape) == 3:
            states = states.view(bs, -1)

        # Ensure agent_qs is the right shape
        agent_qs = agent_qs.view(bs, self.n_agents)

        # Concatenate states and agent_qs
        inputs = torch.cat([states, agent_qs], dim=1)

        q = self.Q(inputs)
        v = self.V(states)

        return q, v


class QTRANAgent:
    def __init__(self, state_size, action_size, num_agents, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.0, epsilon_decay=0.995, decay_method='exponential'):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        # Contain a Q-network for each agent
        self.q_networks = [QTRANQNetwork(
            state_size, action_size) for _ in range(num_agents)]

        # Contain a TARGERT Q-network for each agent to provide stable training by using fixed weights for temporal difference (TD) updates.
        self.target_networks = [QTRANQNetwork(
            state_size, action_size) for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(
                self.q_networks[i].state_dict())

        # self.mixer: Learns how to combine Q-values for joint policy optimization.
        self.mixer = QTRANMixer(state_size * num_agents, num_agents)
        # self.target_mixer: Acts as a stable target for mixer training, initialized with the same weights as self.mixer
        self.target_mixer = QTRANMixer(state_size * num_agents, num_agents)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = optim.Adam(list(self.mixer.parameters()) +
                                    [p for net in self.q_networks for p in net.parameters()],
                                    lr=learning_rate)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_method = decay_method
        self.total_steps = 0
        self.memory = deque(maxlen=10000)

    def update_epsilon(self, episode=None):
        if self.decay_method == 'exponential':
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_method == 'linear':
            self.epsilon = max(self.epsilon_min, self.epsilon_start - (
                self.epsilon_start - self.epsilon_min) * (self.total_steps / 1000000))
        elif self.decay_method == 'cosine':
            self.epsilon = self.epsilon_min + 0.5 * \
                (self.epsilon_start - self.epsilon_min) * \
                (1 + math.cos(math.pi * self.total_steps / 1000000))
        elif self.decay_method == 'step':
            if episode is not None and episode % 100 == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.5)

        self.total_steps += 1

    def act(self, states, sensor_readings):
        actions = []
        for i in range(self.num_agents):
            if random.random() < self.epsilon:
                actions.append(random.randrange(self.action_size))
            else:
                with torch.no_grad():
                    # Converts the agent's state to a PyTorch tensor and processes it with the corresponding Q-network.
                    state = torch.FloatTensor(states[i]).unsqueeze(0)
                    action_values = self.q_networks[i](state).squeeze(0)

                    # Creates a mask to invalidate certain actions based on sensor readings.
                    mask = np.zeros(self.action_size, dtype=float)
                    for j, reading in enumerate(sensor_readings[i]):
                        if reading == 1:
                            mask[j] = float('-inf')
                    # Adds the mask to the Q-values (action_values) to get masked_action_values.
                    masked_action_values = action_values.cpu().numpy() + mask

                    valid_action_indices = np.where(mask == 0)[0]
                    # If no valid actions exist, defaults to the last action (self.action_size - 1).
                    if len(valid_action_indices) == 0:
                        actions.append(self.action_size - 1)
                    # If valid actions exist
                    else:
                        best_action_index = valid_action_indices[np.argmax(
                            masked_action_values[valid_action_indices])]
                        actions.append(best_action_index)
        return actions

    def remember(self, states, actions, reward, next_states, done):
        self.memory.append((states, actions, reward, next_states, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = [torch.FloatTensor(np.array(state)) for state in zip(*states)]
        next_states = [torch.FloatTensor(
            np.array(next_state)) for next_state in zip(*next_states)]
        actions = [torch.LongTensor(action) for action in zip(*actions)]
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = torch.stack([self.q_networks[i](states[i]).gather(1, actions[i].unsqueeze(1)).squeeze(1)
                                        for i in range(self.num_agents)], dim=1)

        state_batch = torch.stack(states, dim=1)
        next_state_batch = torch.stack(next_states, dim=1)

        q_tot, v = self.mixer(state_batch, current_q_values)

        with torch.no_grad():
            next_q_values = torch.stack([self.target_networks[i](next_states[i]).max(1)[
                                        0] for i in range(self.num_agents)], dim=1)
            next_q_tot, next_v = self.target_mixer(
                next_state_batch, next_q_values)

        target_q_tot = rewards + (1 - dones) * self.gamma * next_q_tot

        td_error = nn.MSELoss()(q_tot, target_q_tot)
        opt_loss = nn.MSELoss()(q_tot, current_q_values.sum(dim=1, keepdim=True) - v)
        nopt_loss = torch.mean(torch.clamp(
            q_tot - v - current_q_values.sum(dim=1, keepdim=True), min=0))

        loss = td_error + opt_loss + nopt_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for i in range(self.num_agents):
            self.target_networks[i].load_state_dict(
                self.q_networks[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path):
        torch.save({
            'q_networks_state_dict': [net.state_dict() for net in self.q_networks],
            'target_networks_state_dict': [net.state_dict() for net in self.target_networks],
            'mixer_state_dict': self.mixer.state_dict(),
            'target_mixer_state_dict': self.target_mixer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for i, net in enumerate(self.q_networks):
            net.load_state_dict(checkpoint['q_networks_state_dict'][i])
        for i, net in enumerate(self.target_networks):
            net.load_state_dict(checkpoint['target_networks_state_dict'][i])
        self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        self.target_mixer.load_state_dict(
            checkpoint['target_mixer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def train_qtran(num_episodes=3000, batch_size=32, update_freq=50, save_freq=100,
                epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.998,
                decay_method='exponential'):

    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )

    state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    qtran_agent = QTRANAgent(state_size, action_size, env.num_agents,
                             epsilon_start=epsilon_start, epsilon_min=epsilon_min,
                             epsilon_decay=epsilon_decay, decay_method=decay_method)

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    episode_rewards = []
    best_episode_reward = float('-inf')
    best_episode_actions = None
    best_episode_number = None

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        while not done:
            sensor_readings = env.get_sensor_readings()
            actions = qtran_agent.act(state, sensor_readings)
            next_state, reward, done, actual_actions = env.step(actions)
            qtran_agent.remember(state, actual_actions,
                                 reward, next_state, done)
            qtran_agent.replay(batch_size)
            state = next_state
            total_reward += reward
            episode_actions.append(actual_actions)

        if episode % update_freq == 0:
            qtran_agent.update_target_network()

        episode_rewards.append(total_reward)
        print(
            f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {qtran_agent.epsilon}")

        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            best_episode_actions = episode_actions
            best_episode_number = episode

        if episode % save_freq == 0:
            qtran_agent.save(f'models/qtran_agent_episode_{episode}.pth')

        with open('logs/rewards.txt', 'a') as f:
            f.write(f"{episode},{total_reward}\n")

        # Update epsilon at the end of each episode
        qtran_agent.update_epsilon(episode)

    qtran_agent.save('models/best_qtran_agent.pth')

    save_best_episode(env.initial_positions, best_episode_actions,
                      best_episode_number, best_episode_reward)
    save_final_positions(env, best_episode_actions)
    visualize_and_record_best_strategy(env, best_episode_actions)
    return qtran_agent, best_episode_actions, best_episode_number


# The helper functions save_best_episode, save_final_positions, and visualize_and_record_best_strategy


def save_best_episode(initial_positions, best_episode_actions, best_episode_number, best_episode_reward, filename='qtran_best_strategy.json'):
    action_map = ['forward', 'backward', 'left', 'right', 'stay']

    best_episode = {
        # Convert to int if it's np.int64
        "episode_number": int(best_episode_number),
        # Convert to float if it's np.float64
        "episode_reward": float(best_episode_reward)

    }

    for i in range(len(initial_positions)):
        best_episode[f'agent_{i}'] = {
            'actions': [action_map[action[i]] for action in best_episode_actions],
            'initial_position': initial_positions[i]
        }

    with open(filename, 'w') as f:
        json.dump(best_episode, f, indent=4)

    print(f"Best episode actions and initial positions saved to {filename}")


def save_final_positions(env, best_episode_actions, filename='qtran_final_positions.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()

    for actions in best_episode_actions:
        env.step(actions)

    env.render(
        ax, actions=best_episode_actions[-1], step=len(best_episode_actions)-1)
    plt.title("Final Positions")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final positions saved as {filename}")


def visualize_and_record_best_strategy(env, best_episode_actions, filename='qtran_best_episode.mp4'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()

    # Set up the video writer
    writer = FFMpegWriter(fps=2)

    with writer.saving(fig, filename, dpi=100):
        # Capture the initial state
        ax.clear()
        env.render(ax, actions=None, step=0)
        writer.grab_frame()
        plt.pause(0.1)

        for step, actions in enumerate(best_episode_actions, start=1):
            env.step(actions)
            ax.clear()
            env.render(ax, actions=actions, step=step)
            writer.grab_frame()
            plt.pause(0.1)

    plt.close(fig)
    print(f"Best episode visualization saved as {filename}")


if __name__ == "__main__":
    trained_qtran_agent, best_episode_actions, best_episode_number = train_qtran(
        decay_method='exponential'  # or 'linear', 'cosine', 'step'
    )
    print(f"Best episode: {best_episode_number}")
