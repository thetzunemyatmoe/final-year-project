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
    def __init__(self, actor_state_size, action_size, critic_state_size):
        super().__init__()
        self.actor_state_size = actor_state_size
        self.action_size = action_size

        self.critic_state_size = critic_state_size

        # Policy (Actor)
        self.pi1 = nn.Linear(self.actor_state_size, 128)
        self.pi2 = nn.Linear(128, self.action_size)

        # Value (Critic)
        self.v1 = nn.Linear(self.critic_state_size, 128)
        self.v2 = nn.Linear(128, 1)

        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, partial_state=None, global_state=None):
        logits, value = None, None

        # Policy output
        if partial_state is not None:
            pi1 = F.tanh(self.pi1(partial_state))
            logits = self.pi2(pi1)

        # Value output
        if global_state is not None:
            v1 = F.tanh(self.v1(global_state))
            value = self.v2(v1)  # Single value for the entire state

        return logits, value

    def choose_action(self, state):
        self.eval()
        logits, _ = self.forward(partial_state=state)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, state, actions, value_targets, beta=0.1):
        self.train()
        logits, values = self.forward(state)
        temp_diff = (value_targets - values)
        critic_loss = temp_diff.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        log_prob = m.log_prob(actions)
        actor_loss = -(log_prob.sum(dim=-1) *
                       temp_diff.detach().squeeze())

        total_loss = (critic_loss + actor_loss).mean()
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
        self.local_model = Model(actor_state_size=,
                                 action_size=,
                                 critic_state_size=,
                                 )

        # Env
        self.env = MultiAgentGridEnv(
            grid_file='grid_world_test.json',
            coverage_radius=2
        )

    def run(self):
        total_step = 1
        while self.global_episode.value < MAX_EPISODE:
            # partial_observation and global_state
            state = self.env.reset()

            # Buffers
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
    global_model = Model(actor_state_size=,
                         action_size=,
                         critic_state_size=,
                         )
    global_model.share_memory()

    optimizer = SharedAdam(global_model.parameters(),
                           lr=1e-4, betas=(0.92, 0.999))
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
