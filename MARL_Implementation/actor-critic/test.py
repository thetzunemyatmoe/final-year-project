
#   0: "RIGHT",
#   1: "LEFT ",
#   2: "DOWN ",
#   3: "UP   ",
#   4: "HOVER"

from environment import MultiAgentGridEnv
grid_file = 'grid_world.json'
coverage_radius = 1
max_steps_per_episode = 50
initial_positions = [(0, 1), (4, 3)]

env = MultiAgentGridEnv(
    grid_file=grid_file,
    coverage_radius=coverage_radius,
    initial_positions=initial_positions
)

env.reset()

actions = [[0, 1], [0, 1], [0, 1]]


for action in actions:
    env.step(actions=action, timestep=0, episode=0)
