from environment import MultiAgentGridEnv


grid_file = 'grid_world.json'

env = MultiAgentGridEnv(
    grid_file=grid_file,
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4
)

for i in range(1000):
    env.reset(train=True, seed=i)
