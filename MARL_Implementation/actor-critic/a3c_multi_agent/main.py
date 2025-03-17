import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
from utils import v_wrap, set_init, push_and_pull, record
import matplotlib.pyplot as plt
from shared_adam import SharedAdam
from env import MultiAgentGridEnv
import time

os.environ["OMP_NUM_THREADS"] = "1"

# ENV (For global network)
env = MultiAgentGridEnv(
    grid_file='grid_world_test.json',
    coverage_radius=2
)


BATCH_SIZE = 30
GAMMA = 1
MAX_EPISODE = 7000


state_size = env.get_obs_size()
action_size = env.get_total_actions()
num_agents = env.get_num_agents()


# Neural network
class Model(nn.Module):
    def __init__(self, num_agents, state_size, action_size):
        super().__init__()
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        # Policy network
        self.pi1 = nn.Linear(self.state_size, 128)
        self.pi2 = nn.Linear(128, self.num_agents * self.action_size)

        # Value network
        self.v1 = nn.Linear(self.state_size, 128)
        self.v2 = nn.Linear(128, 1)

        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, state):
        # Policy output
        pi1 = F.tanh(self.pi1(state))
        logits = self.pi2(pi1)
        logits = logits.view(-1, self.num_agents, self.action_size)

        # Value output
        v1 = F.tanh(self.v1(state))
        value = self.v2(v1)  # Single value for the entire state
        return logits, value

    def choose_actions(self, state):
        self.eval()
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1).data
        m = self.distribution(probs)

        return m.sample().numpy().ravel().tolist()  # Returns actions for all agents

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_model, optimizer, global_episode, global_episode_r, res_queue, name, initial_position=None):
        super().__init__()
        self.name = 'w%02i' % name
        # Global network
        self.global_model = global_model

        # Optimizer
        self.optimizer = optimizer

        # Global variable
        self.global_episode = global_episode
        self.global_episode_r = global_episode_r
        self.res_queue = res_queue

        # Local network
        self.local_model = Model(num_agents=num_agents,
                                 state_size=state_size, action_size=action_size)

        # Env
        self.env = MultiAgentGridEnv(
            grid_file='grid_world_test.json',
            coverage_radius=2
        )

    def run(self):
        total_step = 1
        while self.global_episode.value < MAX_EPISODE:
            state = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.

            while True:
                # Forward run
                actions = self.local_model.choose_actions(
                    v_wrap(state[None, :]))
                # Step
                next_state, reward, done = self.env.step(actions)

                episode_reward += reward
                buffer_action.append(actions)
                buffer_state.append(state)
                buffer_reward.append(reward)

                if total_step % BATCH_SIZE == 0 or done:
                    push_and_pull(self.optimizer, self.local_model, self.global_model,
                                  done, next_state, buffer_state, buffer_action, buffer_reward, GAMMA)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:
                        record(self.global_episode, self.global_episode_r,
                               episode_reward, self.res_queue, self.name)
                        print(self.env.poi_coverage_counter)

                        break

                state = next_state
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    global_actor
    gloabl_critic
    global_actor.share_memory()
    global_critic.share_memory()

    global_actor_optimizer = optim.Adam(
        global_actor.parameters(), lr=learning_rate)
    global_critic_optimizer = optim.Adam(
        global_critc.parameters(), lr=learning_rate)

    # global_model = Model(num_agents=num_agents,
    #                      state_size=state_size, action_size=action_size)
    # global_model.share_memory()

    global_episode, global_episode_reward, res_queue = mp.Value(
        'i', 0), mp.Value('d', 0.), mp.Queue()

    initial_positions = []
    workers = [Worker(global_model, optimizer, global_episode,
                      global_episode_reward, res_queue, i) for i in range(mp.cpu_count())]

    [w.start() for w in workers]

    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
