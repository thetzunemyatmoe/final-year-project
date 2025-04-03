from environment import MultiAgentGridEnv


env = MultiAgentGridEnv(
    grid_file='grid_world.json',
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4,
    initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
)


local_obs, state = env.reset()

print(state)

local_obs, reward, done, actual_actions, state = env.step([2, 2, 2, 2])
