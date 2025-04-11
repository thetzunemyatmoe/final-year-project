from environment import MultiAgentGridEnv
from utils import evaluate

GRID_FILE = 'grid_world.json'


env = MultiAgentGridEnv(
    grid_file=GRID_FILE,
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4,
    reward_weight=[0.1, 0.9, 1],
    seed=15
)


evaluate('hello')
