# environment.py
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


class MultiAgentGridEnv:

    def __init__(self, grid_file, coverage_radius, initial_positions=None, seed=None):
        self.grid = self.load_grid(grid_file)
        self.max_fi = 0
        self.grid_height, self.grid_width = self.grid.shape

        # Initialize instance variable
        self.coverage_radius = coverage_radius
        self.num_agents = len(initial_positions)

        self.total_cells = self.grid_height * self.grid_width

        # Calculate new obs_size for local rich observations
        self.central_state_size = 2*self.total_cells
        self.partial_state_size = 2 * (self.coverage_radius*2 + 1)**2

        self.initial_positions = initial_positions

        self.max_step = 100

    def load_grid(self, filename):
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    def reset(self):

        # Sets the agents' positions to their initial positions.
        self.agent_positions = self.initial_positions

        # Reset current step count to zer
        self.current_step = 0

        # Track coverage for each PoI
        self.poi_coverage_counter = np.zeros_like(self.grid)
        self.update_coverage()

        return self.get_local_state(), self.get_central_state()

    # ***********
    # Update MDP after each step
    # ***********

    def update_coverage(self):
        self.coverage_grid = np.zeros_like(self.grid)
        for pos in self.agent_positions:
            self.cover_area(pos)

    def cover_area(self, position):
        x, y = position

        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    self.coverage_grid[ny, nx] = 1
                    self.poi_coverage_counter[ny, nx] += 1

    def step(self, actions):
        # Increment counter
        self.current_step += 1

        # Sensor reading of each UAV
        sensor_readings = self.get_sensor_readings()

        #
        new_positions = []
        actual_actions = []

        # Calculate all new positions
        for i, action in enumerate(actions):
            new_pos = self.get_new_position(self.agent_positions[i], action)
            new_positions.append(new_pos)
            actual_actions.append(action)

        # Penalty for choosing invalid choice
        penalties = [0 for _ in range(self.num_agents)]
        # Validate moves and update positions
        for i, new_pos in enumerate(new_positions):
            if not self.is_valid_move(new_pos, sensor_readings[i], actual_actions[i], new_positions[:i] + new_positions[i+1:]):
                new_positions[i] = self.agent_positions[i]
                actual_actions[i] = 4  # Stay action
                penalties[i] = 5

        print(f'Penalties are {penalties}')
        # Set new postions
        self.agent_positions = new_positions

        # Coverage score on step t-1 (Before updating the coverage state)
        previous_coverage_score = self.poi_coverage_counter / self.max_step

        # Update coverage state and coverage score
        self.update_coverage()

        # Coverage score on step t (After upating the coverage state)
        coverage_score = self.poi_coverage_counter / self.max_step

        # Fairness Index
        total_coverage_score = np.sum(coverage_score)
        squared_coverage_score = np.sum(coverage_score ** 2)
        num_pois = np.count_nonzero(self.grid == 0)
        fairness_index = (total_coverage_score ** 2) / \
            (num_pois * squared_coverage_score)

        self.max_fi = max(self.max_fi, fairness_index)

        # Incremental Coverage
        delta_coverage = np.sum(coverage_score - previous_coverage_score)

        # Incremental Energy Consumption
        delta_energy = 1  # TODO implement the incremental energy consumption

        # Global reward
        reward = fairness_index * delta_coverage

        rewards = []
        # Local reward for each UAV
        for penalty in penalties:
            local_reward = (reward / self.num_agents) - penalty
            rewards.append(local_reward)

        print(rewards)

        # Termianl state?
        done = self.current_step >= self.max_step

        # Return local states, gloabl state, rewards, done
        return self.get_local_state(), self.get_central_state(), rewards, done

    def is_valid_move(self, new_pos, sensor_reading, action, other_new_positions):
        x, y = new_pos
        # Use grid_width and grid_height
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            print(1)
            return False
        if self.grid[y, x] == 1:  # Check for obstacles
            print(2)
            return False
        if new_pos in self.agent_positions or new_pos in other_new_positions:  # Check for other agents
            print(3)
            return False
        # Check sensor readings for specific direction
        if action == 0 and sensor_reading[0] == 1:  # forward
            print(4)
            return False
        elif action == 1 and sensor_reading[1] == 1:  # backward
            print(5)
            return False
        elif action == 2 and sensor_reading[2] == 1:  # left
            print(6)
            return False
        elif action == 3 and sensor_reading[3] == 1:  # right
            print(7)
            return False
        return True

    # Calculate sensor penalty for specific UAV

    def calculate_sensor_penalty(self):
        readings = self.get_sensor_readings()
        total_penalty = 0
        penalty = sum(readings)
        if penalty > 0:
            total_penalty += 1

        return total_penalty

    # Utility function to deal with overlap penalty calculation

    def cover_area_on_grid(self, state, grid):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    grid[ny, nx] += 1  # Increment instead of setting to 1

    # ***********
    # Getter Functions
    # ***********
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

    def get_central_state(self):
        # Normalize the coverage score between 0 and 1
        coverage_scores = self.poi_coverage_counter / self.max_step

        # Binary coverage state (1 if covered in current step, 0 otherwise)
        coverage_state = (self.coverage_grid > 0).astype(int)

        state = np.concatenate([
            coverage_scores.flatten(),  # Coverage score for each PoI
            coverage_state.flatten()])

        return np.array(state)

    def get_local_state(self):
        coverage_scores = self.poi_coverage_counter / self.max_step
        coverage_states = (self.coverage_grid > 0).astype(int)

        local_states = []
        for pos in self.agent_positions:
            x, y = pos

            local_coverage_score = []
            local_coverage_state = []
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        local_coverage_score.append(
                            coverage_scores[ny, nx].item())
                        local_coverage_state.append(
                            coverage_states[ny, nx].item())
                    else:
                        local_coverage_score.append(-1.0)
                        local_coverage_state.append(-1)
            state = np.array(local_coverage_score + local_coverage_state)
            local_states.append(state)

        return local_states

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

    def get_obs_size(self):
        return self.central_state_size

    def get_total_actions(self):
        return 5  # forward, backward, left, right, stay

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
            plt.pause(0.001)
