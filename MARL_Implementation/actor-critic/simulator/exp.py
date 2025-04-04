import gym
import numpy as np
import random
import sys
import pygame


class GymNav(gym.Env):
    def __init__(self, number_of_agents=4, map_size=15):

        self.max_actions = 5
        self.number_of_agents = number_of_agents
        self.map_size = map_size

        # Global info
        self.terminal = False
        self.max_step_limit = 30

        # Position of each agent
        self.pos = []

        # Position of each targer
        self.target_goal = []

        for _ in range(self.number_of_agents):
            self.pos += [[]]

        self.timer = 0

        # GUI
        self.metadata = {'render.modes': ['human']}
        self.screen = None

        # Public GYM variables

        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(self.max_actions) for _ in range(self.number_of_agents)])
        # The Space object corresponding to valid observations
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.number_of_agents, 2))

        # Size of agent observation (10)
        self.agent_observation_space = 2 + 2 * number_of_agents

        # Size of cetral observation (16)
        self.central_observation_space = 2 * number_of_agents + 2 * number_of_agents

        # Size of action space (5)
        self.agent_action_space = self.max_actions

        # A list corresponding to the min and max possible rewards
        self.reward_range = [0, self.number_of_agents]

        print(f'Agent observation space is {self.agent_observation_space}')
        print(f'Central observation space is {self.central_observation_space}')
        print(f'Action space is {self.agent_action_space}')
        print(f'Reward space is {self.reward_range}')

    def render(self, mode='human', close=False):
        cell_width = 50
        cell_width_half = int(cell_width / 2)
        if close:
            pygame.quit()
            self.screen = None
            return

        if mode is not 'human':
            super(GymNav, self).render(mode=mode)
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                [cell_width * self.map_size, cell_width * self.map_size])
            pygame.display.set_caption("Navigation")

        self.screen.fill((255, 255, 255))

        # draw target
        for target in self.target_goal:
            pygame.draw.circle(self.screen, (255, 0, 0),
                               [target[0] * cell_width + cell_width_half,
                                target[1] * cell_width + cell_width_half],
                               int(cell_width_half / 2))

        # Draw agents
        for agent_index, agent in enumerate(self.pos):
            pygame.draw.circle(self.screen, (0, 0, 0),
                               [agent[0] * cell_width + cell_width_half,
                                   agent[1] * cell_width + cell_width_half],
                               cell_width_half)

        pygame.display.flip()

    def reset(self):
        self.target_goal = []

        # Initializing random postions for each agent
        for i in range(self.number_of_agents):
            possible_pos = [random.randint(
                0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            while possible_pos in self.target_goal:
                possible_pos = [random.randint(
                    0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            self.target_goal.append(possible_pos)

        # Initializing random postion for targets
        for i in range(self.number_of_agents):
            possible_pos = [random.randint(
                0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            while possible_pos in self.pos or possible_pos in self.target_goal:
                possible_pos = [random.randint(
                    0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            self.pos[i] = possible_pos

        self.timer = 0

        return self.get_state(), self.get_state_central()

    def step(self, action):
        reward = 0
        for i, action in enumerate(action):
            if action == 0:
                mov = [0, 1]
            elif action == 1:
                mov = [0, -1]
            elif action == 2:
                mov = [1, 0]
            elif action == 3:
                mov = [-1, 0]
            else:
                mov = [0, 0]
            self.pos[i][0] = (self.pos[i][0] + mov[0]) % self.map_size
            self.pos[i][1] = (self.pos[i][1] + mov[1]) % self.map_size

        for obstacle in self.target_goal:
            min_dist = self.map_size
            for i in range(self.number_of_agents):
                min_dist = min(min_dist, dist(self.pos[i], obstacle))
            reward += (self.map_size / 2 - min_dist) / (self.map_size / 2)

        terminal = self.timer >= self.max_step_limit or reward == self.number_of_agents
        if not terminal:
            reward = 0

        self.timer += 1

        return self.get_state(), reward, terminal,  self.get_state_central()

    # Central state
    def get_state_central(self):
        central_state = []
        for x in self.pos:
            central_state.extend(x)
        for x in self.target_goal:
            central_state.extend(x)

        return [central_state for _ in range(self.number_of_agents)]

    # Agent's observation
    def get_state(self):

        obs = []
        for index, pos in enumerate(self.pos):
            state = list(pos)
            for x in self.target_goal:
                state.extend(x)
            obs.append(state)

        return obs

    def close(self):
        return

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]


# manhattan distance
def dist(a, b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])
