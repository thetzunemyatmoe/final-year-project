# environment.py
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
import random
import time


class MultiAgentGridEnv:

    def __init__(self, grid_file, coverage_radius, max_steps_per_episode, num_agents, reward_weight=None, seed=None, initial_positions=None):

        if seed is not None:
            random.seed(seed)

        # Default weight for reward
        if reward_weight is None:
            self.reward_weight = {
                'total area weight': 1.0,
                'overlap weight': 1.0,
                'energy weight': 1.0
            }
        else:
            self.reward_weight = reward_weight

        self.seed = seed
        # Grid and its properties
        self.grid = self.load_grid(grid_file)
        self.grid_height, self.grid_width = self.grid.shape
        self.total_cells_to_cover = np.sum(self.grid == 0)  # For termination

        # Initialize instance variable
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.num_agents = num_agents

        if initial_positions is not None:
            self.initial_positions = initial_positions
        else:
            self.initial_positions = self.initialize_position()

        # Calculate new obs_size for local rich observations
        self.obs_size = (
            2 +  # Agent's own position (x, y)  ###
            4 +  # Sensor readings
            1 +  # Current time step
            # Local view of coverage grid and the map
            (2*coverage_radius + 1)**2 * 2 +
            (num_agents - 1) * 2  # Relative positions of other agents (x, y)
        )

        # Calcualte size of the state
        self.state_size = (2 * self.num_agents + (self.grid_height *
                           self.grid_width) + 2*(self.num_agents * (self.num_agents - 1)) + 1)

    def load_grid(self, filename):
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    def initialize_position(self):
        initial_positions = []
        x = random.randint(2, self.grid_height-3)
        y = random.randint(2, self.grid_width-3)

        initial_positions.append((x, y))
        initial_positions.append((x+1, y))
        initial_positions.append((x, y+1))
        initial_positions.append((x+1, y+1))

        # print(initial_positions)
        return initial_positions

    def reset(self, train=None):
        self.done = False

        # Metric
        self.energy_usage = []

        if train is not None:
            seed = (train % 50)
            random.seed(seed)
            self.initial_positions = self.initialize_position()

        # Sets the agents' positions to their initial positions.
        self.agent_positions = list(self.initial_positions)

        # Reset current step count to zero
        self.current_step = 0

        # Update the coverage grid based on agent's initial position
        self.coverage_grid = np.zeros_like(self.grid)
        self.update_coverage()

        return self.get_observations(), self.get_state()

    def update_coverage(self):
        for pos in self.agent_positions:
            self.cover_area(pos)

    def step(self, actions):
        self.current_step += 1
        new_positions = []
        actual_actions = []
        sensor_readings = self.get_sensor_readings()

        # First, calculate all new positions
        for i, action in enumerate(actions):
            new_pos = self.get_new_position(self.agent_positions[i], action)
            new_positions.append(new_pos)
            actual_actions.append(action)

        # Then, validate moves and update positions
        invalid_penalty = 0
        for i, new_pos in enumerate(new_positions):
            if not self.is_valid_move(new_pos, sensor_readings[i], actual_actions[i], new_positions[:i] + new_positions[i+1:]):
                new_positions[i] = self.agent_positions[i]
                actual_actions[i] = 4  # Stay action
                invalid_penalty += 1

        # Previous coverage
        self.prev_total_area = np.sum(self.coverage_grid)

        # Update coverage map
        self.agent_positions = new_positions
        self.update_coverage()

        self.done = self.current_step >= self.max_steps_per_episode

        # Global Reward
        global_reward = self.calculate_global_reward(
            actions=actual_actions) - invalid_penalty

        self.energy_usage.append(self.energy_penalty)

        # print(
        #     f'{self.current_step} | {np.sum(self.coverage_grid)} | {self.total_cells_to_cover}')

        return self.get_observations(), global_reward, self.done, actual_actions, self.get_state()

    def is_valid_move(self, new_pos, sensor_reading, action, other_new_positions):
        x, y = new_pos
        # Use grid_width and grid_height
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False
        if self.grid[y, x] == 1:  # Check for obstacles
            return False
        if new_pos in self.agent_positions or new_pos in other_new_positions:  # Check for other agents
            return False
        # Check sensor readings for specific direction
        if action == 0 and sensor_reading[0] == 1:  # forward
            return False
        elif action == 1 and sensor_reading[1] == 1:  # backward
            return False
        elif action == 2 and sensor_reading[2] == 1:  # left
            return False
        elif action == 3 and sensor_reading[3] == 1:  # right
            return False
        return True

    def get_new_position(self, position, action):
        x, y = position
        if action == 0:  # forward (positive x)
            return (min(x + 1, self.grid_width - 1), y)
        elif action == 1:  # backward (negative x)
            return (max(x - 1, 0), y)
        elif action == 2:  # left (positive y)
            return (x, min(y + 1, self.grid_height - 1))
        elif action == 3:  # right (negative y)
            return (x, max(y - 1, 0))
        else:  # stay
            return (x, y)

    def cover_area(self, state):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    self.coverage_grid[ny, nx] = 1

    # ***********
    # Reward Calculation
    # ***********

    def calculate_global_reward(self, actions):

        # Rewards
        self.total_area_gain = np.sum(
            self.coverage_grid) - self.prev_total_area

        # Penalties
        self.overlap_penalty = self.calculate_overlap()
        self.sensor_penalty = self.calculate_sensor_penalty() * \
            ((1 + 2*self.coverage_radius)**2)
        self.energy_penalty = self.calculate_energy_penalty(actions)

        reward = (
            self.reward_weight['total area weight'] * self.total_area_gain
            - self.reward_weight['overlap weight'] * self.overlap_penalty
            - 0.5 * self.sensor_penalty
            - self.reward_weight['energy weight'] * self.energy_penalty
        )
        return reward

    def calculate_energy_penalty(self, actions):

        total_penalty = 0
        for action in actions:
            if action > 3:
                total_penalty += 1
            else:
                total_penalty += 3

        return total_penalty

    def calculate_sensor_penalty(self):
        sensor_readings = self.get_sensor_readings()
        total_penalty = 0
        for readings in sensor_readings:
            # Sum up the number of 'blocked' directions (1's in the sensor reading)
            penalty = sum(readings)
            if penalty > 0:
                total_penalty += 1

        return total_penalty

    def calculate_overlap(self):
        overlap_grid = np.zeros_like(self.coverage_grid)
        for pos in self.agent_positions:
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)
            overlap_grid += temp_grid

        overlap_counts = overlap_grid[overlap_grid > 1] - 1
        weighted_overlap = np.sum(overlap_counts)
        return weighted_overlap

    def cover_area_on_grid(self, state, grid):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    grid[ny, nx] += 1  # Increment instead of setting to 1

    # ***********
    # Get methods
    # ***********

    def get_observations(self):
        observations = []
        sensor_readings = self.get_sensor_readings()

        for i, pos in enumerate(self.agent_positions):
            x, y = pos
            obs = [
                x, y,  # Agent's own position (x, y)
                *sensor_readings[i],  # Sensor readings
                self.current_step,  # Current time step
            ]

            # Local view of coverage and obstacles
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        obs.extend([
                            self.coverage_grid[ny, nx],
                            self.grid[ny, nx]
                        ])
                    else:
                        # Treat out-of-bounds as uncovered and obstacle
                        obs.extend([0, 1])

            # Relative positions of nearby agents
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:
                    ox, oy = other_pos
                    if abs(x - ox) <= self.coverage_radius and abs(y - oy) <= self.coverage_radius:
                        obs.extend([ox - x, oy - y])
                    else:
                        # Indicate agent is out of local view
                        obs.extend([self.coverage_radius * 4,
                                   self.coverage_radius * 4])

            observations.append(np.array(obs, dtype=np.float32))

        return observations

    def get_state(self):

        state = []

        # Position of each agent
        for pos in self.agent_positions:
            state.append(pos[0])
            state.append(pos[1])

        coverage_map = np.where(self.grid == 1, -1, 0)
        coverage_map = np.where(self.coverage_grid == 1, 1, coverage_map)

        # Coveage map (0: Uncovered, 1: Covered, -1: Obstacles)
        state.extend(coverage_map.flatten().tolist())

        # Relative position
        for i in range(len(self.agent_positions)):
            xi, yi = self.agent_positions[i]
            for j in range(len(self.agent_positions)):
                if i != j:
                    xj, yj = self.agent_positions[j]
                    dx = xj - xi
                    dy = yj - yi
                    state.append(dx)
                    state.append(dy)

        state.append(self.current_step)

        return state

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_total_actions(self):
        return 5  # forward, backward, left, right, stay

    def get_sensor_readings(self):
        readings = []
        for pos in self.agent_positions:
            x, y = pos
            reading = [
                1 if x == self.grid_width -
                # forward
                1 or self.grid[y, x + 1] == 1 or (x + 1, y) in self.agent_positions else 0,
                # backward
                1 if x == 0 or self.grid[y, x - 1] == 1 or (
                    x - 1, y) in self.agent_positions else 0,
                1 if y == self.grid_height -
                # left
                1 or self.grid[y + 1, x] == 1 or (x, y + 1) in self.agent_positions else 0,
                # right
                1 if y == 0 or self.grid[y - 1, x] == 1 or (
                    x, y - 1) in self.agent_positions else 0
            ]
            readings.append(reading)
        return readings

    # ***********
    # Utility Function
    # ***********

    def get_metrics(self):

        total_cells = self.total_cells_to_cover
        total_cells_covered = np.sum(self.coverage_grid)

        total_steps = self.current_step
        total_energy = sum(self.energy_usage)
        return {
            "Total Energy Usage": total_energy,
            "Step Taken": total_steps,
            "Average Energy": total_energy / total_steps,
            f"Total cells covered (Out of {total_cells})":  int(total_cells_covered),
            "Coverage Rate": f'{(total_cells_covered / total_cells) * 100} %'
        }

    def render(self, ax=None, actions=None, step=None, return_rgb=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        ax.clear()
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)

        # Draw the grid and obstacles
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j] == 1:  # Obstacles are black
                    rect = plt.Rectangle((j, i), 1, 1, color='black')
                    ax.add_patch(rect)

        # Define consistent colors for 10 agents
        agent_colors = ['red', 'blue', 'green', 'yellow',
                        'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']

        # Draw the coverage area and agents
        for idx, pos in enumerate(self.agent_positions):
            x, y = pos
            agent_color = agent_colors[idx % len(agent_colors)]

            # Draw coverage area
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                        rect = plt.Rectangle(
                            (nx, ny), 1, 1, color=agent_color, alpha=0.3)
                        ax.add_patch(rect)

            # Draw the agent
            rect = plt.Rectangle((x, y), 1, 1, color=agent_color)
            ax.add_patch(rect)

            # Add agent number
            ax.text(x + 0.5, y + 0.5, str(idx + 1), color='black',
                    ha='center', va='center', fontweight='bold')

        # Display sensor readings
        sensor_readings = self.get_sensor_readings()
        for agent_idx, pos in enumerate(self.agent_positions):
            readings = sensor_readings[agent_idx]
            ax.text(pos[0] + 0.5, pos[1] - 0.3,
                    f'{readings}', color='red', ha='center', va='center', fontsize=8)

        ax.grid(True)
        if actions is not None:
            action_texts = ['forward', 'backward', 'left', 'right', 'stay']
            action_display = ' | '.join(
                [f"Agent {i+1}: {action_texts[action]}" for i, action in enumerate(actions)])
            title = f'{action_display}'
            if step is not None:
                title += f' || Step: {step}'
            ax.set_title(title, fontsize=10)

        if return_rgb:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        else:
            plt.draw()
