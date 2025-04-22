from IA2CC import IA2CC
from environment import MultiAgentGridEnv
from utils import evaluate

GRID_FILE = 'grid_world.json'

env = MultiAgentGridEnv(
    grid_file=GRID_FILE,
    coverage_radius=4,
    max_steps_per_episode=50,
    num_agents=4,
)


# NN pararmeters
critic_input_size = env.get_state_size()
actor_input_size = env.get_obs_size()
actor_output_size = env.get_total_actions()

model = IA2CC(actor_input_size=actor_input_size,
              actor_output_size=actor_output_size,
              critic_input_size=critic_input_size,
              num_agents=env.num_agents)

model.load_actors('model/main')

evaluate(model)
