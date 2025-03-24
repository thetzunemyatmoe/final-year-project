
from environment import MultiAgentGridEnv
from IA2CC import IA2CC


max_episode = 1000
env = MultiAgentGridEnv(
    grid_file='grid_world.json',
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4,
    initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
)


# NN pararmeters
critic_input_size = env.num_agents * env.get_obs_size()
actor_input_size = env.get_obs_size()
actor_output_size = env.get_total_actions()

ia2cc = IA2CC(actor_input_size=actor_input_size,
              actor_output_size=actor_output_size, critic_input_size=critic_input_size, num_agents=env.num_agents)

episode_reward = 0
episodes_reward = []
joint_observations = env.reset()
episode = 0
while episode < max_episode:
    actions = ia2cc.act(joint_observations)
    next_joint_observation, reward, done, _ = env.step(actions)

    # agents.memory.reward.append(reward)
    # for i in range(agent_num):
    #     agents.memory.done[i].append(done_n[i])

    episode_reward += reward

    joint_observations = next_joint_observation

    if done:
        episodes_reward.append(episode_reward)
        episode_reward = 0

        episode += 1

        obs = env.reset()

        if episode % 10 == 0:
            # Update
            pass

        if episode % 100 == 0:
            print(
                f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
