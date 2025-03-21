from new_env import MultiAgentGridEnv


env = MultiAgentGridEnv(grid_file="grid_world.json", coverage_radius=1, max_steps_per_episode=200,
                        num_agents=1, initial_positions=[(5, 5)])


print(env.get_obs_size())
state = env.reset()


print(state)

state, reward, done, actual_actions = env.step([0])

print(state)


state, reward, done, actual_actions = env.step([0])

print(state)


state, reward, done, actual_actions = env.step([3])

print(state)
