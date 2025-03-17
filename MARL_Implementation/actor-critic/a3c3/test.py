from env import MultiAgentGridEnv
import random

env = MultiAgentGridEnv(
    grid_file='grid_world_test.json',
    coverage_radius=1,
    initial_positions=[(0, 0), (0, 1)]
)


env.reset()
env.step([4, 4])


print(env.poi_coverage_counter)
