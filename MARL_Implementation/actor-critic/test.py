from environment import MultiAgentGridEnv
env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=4,
        max_steps_per_episode=50,
        num_agents=4,
        initial_positions=[(25, 25), (1, 1), (2, 2), (3, 3)]
    )

print(env.get_observations()[0])