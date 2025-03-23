from env import MultiAgentGridEnv


env = MultiAgentGridEnv(
    grid_file='grid_world_test.json',
    coverage_radius=2,
    initial_positions=[(0, 0)]
)


env.reset()
