from global_env import MultiAgentGridEnv

env = MultiAgentGridEnv(grid_file="grid_world.json", coverage_radius=1, max_steps_per_episode=200,
                        num_agents=1, initial_positions=[(0, 0)])

state = env.reset()
print(state)
print('Exuecute')

for i in range(5):

    state, reward, done, acttual_action = env.step([0])

    print(state)
    print(reward)
    print(done)
    print(acttual_action)
    print('------------------')
