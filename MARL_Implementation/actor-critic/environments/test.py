from partial_environent import MultiAgentGridEnv

env = MultiAgentGridEnv(grid_file='grid_world_test.json',
                        coverage_radius=2, initial_positions=[(0, 0), (0, 1)])

print(f'Partial state has size {env.partial_state_size}')
print(f'Central state has size {env.central_state_size}')


partial_states, central_state = env.reset()


for index, partial_states in enumerate(partial_states):
    print(
        f'Partial state {env.agent_positions[index]} --> {partial_states}')
print(central_state)


partial_states, central_state, rewards, done = env.step([2, 2])

for index, partial_state in enumerate(partial_states):
    print(
        f'Partial state {env.agent_positions[index]} --> {partial_state}')
print(central_state)
print(f'Reward is {rewards}')
