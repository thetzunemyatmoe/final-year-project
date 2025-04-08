from utils import load_actors, save_best_episode, save_final_positions, visualize_and_record_best_strategy
from Actor import Actor
from environment import MultiAgentGridEnv
import torch


def evaluate_multi_agent(actors, env, num_episodes=10):
    num_agents = len(actors)
    obs, _ = env.reset()
    done = False
    rewards = 0.0
    episode_actions = []

    while not done:
        actions = []
        for i in range(num_agents):
            actor = actors[i]
            obvs = obs[i]
            action, _, _ = actor(obvs)
            # action = actors[i](obs[i])
            actions.append(action)
        print(actions)
        obs, reward, done, actions, _ = env.step(actions)

        rewards += reward
        episode_actions.append(actions)

    print(rewards)
    # save_best_episode(env.initial_positions, best_episode_actions,
    #                   best_episode_number, best_episode_reward)
    # save_final_positions(env, best_episode_actions)
    visualize_and_record_best_strategy(env, episode_actions)


GRID_FILE = 'grid_world.json'
# env
env = MultiAgentGridEnv(
    grid_file=GRID_FILE,
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4
)


# Load models
actors = load_actors(
    Actor, 4, input_size=env.get_obs_size(), output_size=env.get_total_actions(), path="./model/learning_rate/0.0001&0.001/")


evaluate_multi_agent(actors, env)
