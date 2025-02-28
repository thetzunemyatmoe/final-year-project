# environment.py
import numpy as np
import json
import matplotlib.pyplot as plt


class MultiAgentGridEnv:

    def __init__(self, grid_file, coverage_radius, max_steps_per_episode, initial_positions, reward_type='global'):
        # Load the grid from the JSON file
        self.grid = self.load_grid(grid_file)

        # Obtain height and width (shape function of an np array)
        self.grid_height, self.grid_width = self.grid.shape

        # Initialize instance variable
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.num_agents = len(initial_positions)
        self.initial_positions = initial_positions
        self.reward_type = reward_type

        # Calculate new obs_size for local rich observations
        self.obs_size = (
            2 +  # Agent's own position (x, y)  ###
            4 +  # Sensor readings
            1 +  # Current time step
            # Local view of coverage grid and the map
            (2*coverage_radius + 1)**2 * 2 +
            (self.num_agents - 1) * 2  # Relative positions of other agents (x, y)
        )

        # Reset the environment to initial state
        self.reset()

    def load_grid(self, filename):
        """
        Loads a 2D grid from a JSON file and converts it to a NumPy array.

        Args:
            filename (str): The path to the JSON file containing the 2D grid data.

        Returns:
            numpy.ndarray: A NumPy array representing the loaded 2D grid.
        """
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    def reset(self):

        # Sets the agents' positions to their initial positions.
        self.agent_positions = list(self.initial_positions)

        # Resets the coverage grid to zero (No ares have been covered)
        # self.coverage_grid = np.zeros_like(self.grid)

        # Reset current step count to zero
        self.current_step = 0

        # Reward tracking
        self.reward_track = {i: {"total_area": 0, "overlap": 0}
                             for i in range(self.num_agents)}

        # Update the coverage grid based on agent's initial position
        self.update_coverage()
        self.calculate_overlap()

        return self.get_observations()

    def update_coverage(self):
        """
            Updates the coverage grid based on the current positions of all agents
        """
        # Resets the coverage grid to zero (No ares have been covered)
        self.coverage_grid = np.zeros_like(self.grid)

        # Update coverage for each agent based on their current position
        for index, pos in enumerate(self.agent_positions):
            # Reset the total area for each agent after each step
            self.reward_track[index]['total_area'] = 0
            self.cover_area(index, pos)

    # Manipulate the coverage grid

    def cover_area(self, index, position):

        # Extract the x and y coordinates of the agent's position
        x, y = position
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                # Mark the cell when it's valid (empty and within range)
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    self.coverage_grid[ny, nx] = 1
                    self.reward_track[index]["total_area"] += 1

    def step(self, actions):
        # print(actions[1])
        """
            Executes a single step in the environment based on the agents' actions.
        """
        # Increment the step count
        self.current_step += 1

        #
        new_positions = []
        actual_actions = []

        # Get sensor readings for each
        sensor_readings = self.get_sensor_readings()

        # First, calculate all new positions given the actios
        for i, action in enumerate(actions):
            new_pos = self.get_new_position(self.agent_positions[i], action)
            new_positions.append(new_pos)
            actual_actions.append(action)

        # Then, validate moves and update positions
        for i, new_pos in enumerate(new_positions):
            # If the new position (after taking actions) is not valid, revert back to previous position
            if not self.is_valid_move(new_pos, sensor_readings[i], actual_actions[i], new_positions[:i] + new_positions[i+1:]):
                new_positions[i] = self.agent_positions[i]
                actual_actions[i] = 4  # Stay action (Hover)

        # Update the evnrionment with the new validated positions
        self.agent_positions = new_positions

        # Update the coverage gird based on the new validated positions
        self.update_coverage()
        self.calculate_overlap()

        # Calculate the global reward for the current step
        # global_reward = self.calculate_global_reward()
        rewards = []
        for index in range(self.num_agents):
            rewards.append(self.calculate_individual_reward(index))

        # Check if the maximun allowed steps reached
        done = self.current_step >= self.max_steps_per_episode
        return self.get_observations(), rewards, done, actual_actions

    def get_new_position(self, position, action):
        x, y = position
        print(f'Action receive by get_new_positon {action}')
        if action == 0:  # X - RIGHT
            return (min(x + 1, self.grid_width - 1), y)
        elif action == 1:  # X - LEFT
            return (max(x - 1, 0), y)
        elif action == 2:  # Y - DOWN
            return (x, min(y + 1, self.grid_height - 1))
        elif action == 3:  # Y - UP
            return (x, max(y - 1, 0))
        else:  # stay
            return (x, y)

    def is_valid_move(self, new_pos, sensor_reading, action, other_new_positions):
        x, y = new_pos
        # Check wthether the move is within the grid
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False

        # Check for obstacles
        if self.grid[y, x] == 1:
            return False

        # Check for other agents
        if new_pos in self.agent_positions or new_pos in other_new_positions:
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

    # ***********
    # Reward Calculation
    # ***********

    # Reward for individual agent
    def calculate_individual_reward(self, index):
        self.total_area = self.reward_track[index]["total_area"]
        self.overlap = self.reward_track[index]["overlap"]

        # self.sensor_1s = self.calculate_sensor_penalty()
        # self.sensor_penalty = self.sensor_1s * \
        #     ((1 + 2*self.coverage_radius)**2)

        reward = (
            self.total_area
            - (0.75) * self.overlap
        )
        return reward

    def calculate_sensor_penalty(self):
        sensor_readings = self.get_sensor_readings()
        total_penalty = 0
        for readings in sensor_readings:
            # Sum up the number of 'blocked' directions (1's in the sensor reading)
            penalty = sum(readings)
            if penalty > 0:
                total_penalty += 1

        return total_penalty

    # def areas_overlap(self, pos1, pos2):
    #     x1, y1 = pos1
    #     x2, y2 = pos2
    #     return abs(x1 - x2) <= 2 * self.coverage_radius and abs(y1 - y2) <= 2 * self.coverage_radius

    def calculate_overlap(self):
        overlap_grid = np.zeros_like(self.coverage_grid)
        for pos in self.agent_positions:
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)
            overlap_grid += temp_grid

        # Now, calculate overlap contribution for each UAV
        for index, pos in enumerate(self.agent_positions):
            self.reward_track[index]["overlap"] = 0
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)

            # Compute overlap introduced by this UAV
            # Only count where overlap happens
            overlap_counts = (overlap_grid > 1) * temp_grid
            self.reward_track[index]["overlap"] = int(np.sum(overlap_counts))

    # Utility function to deal with overlap penalty calculation
    def cover_area_on_grid(self, state, grid):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    grid[ny, nx] += 1  # Increment instead of setting to 1

    # ***********
    # Reward Calculation end
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

    def get_obs_size(self):
        return self.obs_size

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

    # Can be useful for debugging

    def get_metrics(self):
        return {
            "Total Area": self.total_area,
            "Overlap Penalty": self.overlap_penalty,
            "Connectivity Penalty": self.connectivity_penalty,
            "Hole Penalty": self.hole_penalty,
            "Number of Components": self.num_components,
            "Number of Holes": len(self.find_chordless_cycles(self.build_graph())),
            "Reward": self.total_area - self.overlap_penalty - self.connectivity_penalty - self.hole_penalty
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
            plt.pause(0.001)
