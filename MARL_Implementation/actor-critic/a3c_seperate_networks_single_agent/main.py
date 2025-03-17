import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import os
from utils import v_wrap, push_and_pull, record
import matplotlib.pyplot as plt
from env import MultiAgentGridEnv
from shared_adam import SharedAdam
import time

os.environ["OMP_NUM_THREADS"] = "1"

# ENV (For global network)
env = MultiAgentGridEnv(
    grid_file='grid_world_test.json',
    coverage_radius=3,
    seed=42
)


BATCH_SIZE = 50
GAMMA = 1
MAX_EPISODE = 500
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5


state_size = env.get_obs_size()
action_size = env.get_total_actions()


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size)
        )
        self.distribution = torch.distributions.Categorical

    def forward(self, input):
        return self.net(input)

    def choose_action(self, state):
        logits = self.forward(input=state)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(probs=prob)
        return m.sample().numpy()[0]


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        return self.net(input)


class Worker(mp.Process):
    def __init__(self, global_actor, global_critic, optimizer, global_episode, global_episode_r, res_queue, name, initial_position=None):
        super().__init__()
        self.name = 'w%02i' % name
        # Global networks
        self.global_actor = global_actor
        self.global_critic = global_critic

        # Optimizer
        # self.actor_optimizer = actor_optimizer
        # self.critic_optimizer = critic_optimizer
        self.optimizer = optimizer
        # Global variable
        self.global_episode = global_episode
        self.global_episode_r = global_episode_r
        self.res_queue = res_queue

        # Local networks
        self.local_actor = Actor(
            input_size=state_size, output_size=action_size)
        self.local_critic = Critic(input_size=state_size)

        # Env
        self.env = MultiAgentGridEnv(
            grid_file='grid_world_test.json',
            coverage_radius=3,
            seed=42
        )

    def run(self):
        total_step = 1
        while self.global_episode.value < MAX_EPISODE:
            state = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.0

            while True:
                action = self.local_actor.choose_action(v_wrap(state[None, :]))
                next_state, reward, done = self.env.step(action)

                episode_reward += reward
                buffer_action.append(action)
                buffer_state.append(state)
                buffer_reward.append(reward)

                if total_step % BATCH_SIZE == 0 or done:
                    push_and_pull(self.local_actor, self.local_critic, self.global_actor, self.global_critic, self.optimizer,
                                  done, next_state, buffer_state, buffer_action, buffer_reward, GAMMA)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:
                        record(self.global_episode, self.global_episode_r,
                               episode_reward, self.res_queue, self.name)

                        break

                state = next_state
                total_step += 1

        print(self.env.poi_coverage_counter)
        self.res_queue.put(None)


if __name__ == "__main__":
    global_actor = Actor(input_size=state_size, output_size=action_size)
    global_critic = Critic(input_size=state_size)
    global_actor.share_memory()
    global_critic.share_memory()

    optimizer = SharedAdam(list(global_actor.parameters()) +
                           list(global_critic.parameters()))

    global_episode, global_episode_reward, res_queue = mp.Value(
        'i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [Worker(global_actor=global_actor, global_critic=global_critic, optimizer=optimizer,
                      global_episode=global_episode, global_episode_r=global_episode_reward,
                      res_queue=res_queue, name=i) for i in range(mp.cpu_count())]

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
