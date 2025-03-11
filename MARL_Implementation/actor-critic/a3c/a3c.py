import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
from utils import v_wrap, set_init, push_and_pull, record
import matplotlib.pyplot as plt
from shared_adam import SharedAdam

import gym

# os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EPISODE = 3000

# ENV
env = gym.make('CartPole-v1')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


# Neural network class
class Net(nn.Module):
    def __init__(self, p_state_size, action_size):
        super().__init__()
        # Policy
        self.p_state_size = p_state_size
        self.action_size = action_size

        # Value

        self.pi1 = nn.Linear(self.p_state_size, 128)
        self.pi2 = nn.Linear(128, self.action_size)
        self.v1 = nn.Linear(self.p_state_size, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, state):
        # Policy
        pi1 = torch.tanh(self.pi1(state))
        logits = self.pi2(pi1)

        # Value
        v1 = torch.tanh(self.v1(state))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()  # Evluation mode
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, state, action, value_t):
        self.train()
        logits, values = self.forward(state)
        td = value_t - values
        critic_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(action) * td.detach().squeeze()
        actor_loss = -exp_v
        total_loss = (critic_loss + actor_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_r, res_queue, name):
        super().__init__()
        self.name = 'w%02i' % name
        # Global network
        self.global_net = global_net

        # Optimizer
        self.optimizer = optimizer

        # Global variable
        self.global_episode = global_episode
        self.global_episode_r = global_episode_r
        self.res_queue = res_queue

        # Local network
        self.local_network = Net(N_S, N_A)

        # Env
        self.env = gym.make('CartPole-v1').unwrapped

    def run(self):
        total_step = 1
        while self.global_episode.value < MAX_EPISODE:
            state, _ = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.

            while True:
                # if self.name == 'w00':
                #     self.env.render()
                action = self.local_network.choose_action(
                    v_wrap(state[None, :]))
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = terminated or truncated

                if done:
                    reward = -1

                episode_reward += reward
                buffer_action.append(action)
                buffer_state.append(state)
                buffer_reward.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.optimizer, self.local_network, self.global_net,
                                  done, next_state, buffer_state, buffer_action, buffer_reward, GAMMA)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:
                        record(self.global_episode, self.global_episode_r,
                               episode_reward, self.res_queue, self.name)
                        break

                state = next_state
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    global_net = Net(N_S, N_A)
    global_net.share_memory()

    optimizer = SharedAdam(global_net.parameters(),
                           lr=1e-4, betas=(0.92, 0.999))
    global_episode, global_episode_reward, res_queue = mp.Value(
        'i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [Worker(global_net, optimizer, global_episode,
                      global_episode_reward, res_queue, i) for i in range(mp.cpu_count())]

    [w.start() for w in workers]

    res = []                    # record episode reward to plot
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
