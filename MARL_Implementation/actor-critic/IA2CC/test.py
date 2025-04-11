from utils import load_actors, save_best_episode, save_final_positions, visualize_trajectory
from Actor import Actor
from new_env import MultiAgentGridEnv
import torch


def evaluate_multi_agent(actors, env):
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
            actions.append(action)
        obs, reward, done, actions, _ = env.step(actions)

        rewards += reward
        episode_actions.append(actions)

    return rewards, env.get_metrics()


GRID_FILE = 'grid_world.json'
# env
env = MultiAgentGridEnv(
    grid_file=GRID_FILE,
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4,
    seed=31
)


# Load models
actors = load_actors(
    Actor, 4, input_size=env.get_obs_size(), output_size=env.get_total_actions(), path="./model/learning_rate/0.0001&0.001/")


evaluate_multi_agent(actors, env)
