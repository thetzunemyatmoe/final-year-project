
from extras.environment import MultiAgentGridEnv
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

# Hyperparameter
gamma = 1
step = 0
update_interval = 30
# Buffers for 30-step rollout
obs_buffer = []
next_obs_buffer = []
log_probs_buffer = []
entropies_buffer = []
rewards_buffer = []

while episode < max_episode:

    actions, log_probs, entropies = ia2cc.act(joint_observations)
    next_joint_observations, reward, done, _ = env.step(actions)

    episode_reward += reward

    # ia2cc.compute_loss(
    #     reward, joint_observations, next_joint_observations, log_probs, entropies
    # )

    # Store in buffer
    obs_buffer.append(joint_observations)
    next_obs_buffer.append(next_joint_observations)
    log_probs_buffer.append(log_probs)
    entropies_buffer.append(entropies)
    rewards_buffer.append(reward)

    joint_observations = next_joint_observations
    step += 1

    if done:
        if episode % 10 == 0:
            print(f"Episode {episode} ended with reward {episode_reward}")
            print(env.agent_positions)
        episode += 1
        joint_observations = env.reset()
        episode_reward = 0
        i = 0
