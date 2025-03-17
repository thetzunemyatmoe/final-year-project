from env import MultiAgentGridEnv
import random

env = MultiAgentGridEnv(
    grid_file='grid_world_test.json',
    coverage_radius=1,
    initial_positions=[(0, 0), (0, 1)]
)


env.reset()

for i in range(10000):
    action1 = random.randint(0, 3)
    action2 = random.randint(0, 3)
    env.step([action1, action2])


print(env.poi_coverage_counter)
