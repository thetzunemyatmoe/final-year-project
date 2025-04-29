import os
import json
import numpy as np

# Path to the base 'evaluate' directory
base_dir = 'evaluate'

# List to store all extracted 'Metic' data
all_metics = []

metrics = {}


# Walk through each seed_* directory
for seed_folder in os.listdir(base_dir):
    seed_path = os.path.join(base_dir, seed_folder)

    # Seed number
    _, seed = seed_folder.split("_")

    if os.path.isdir(seed_path) and seed_folder.startswith('seed_'):
        rewardweight_path = os.path.join(seed_path, 'rewardweight')

        if os.path.exists(rewardweight_path):
            for config_folder in os.listdir(rewardweight_path):

                # Reward weight configuration
                m = len(config_folder)
                config = config_folder[m-1]

                if config not in metrics:
                    metrics[config] = {}
                if seed not in metrics[config]:
                    metrics[config][seed] = {}

                config_path = os.path.join(rewardweight_path, config_folder)
                if os.path.isdir(config_path) and config_folder.startswith('config'):
                    stats_file = os.path.join(config_path, 'statistics.json')
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            data = json.load(f)
                            if 'Metic' in data:
                                all_metics.append(data['Metic'])
                                metrics[config][seed] = data['Metic']


for config in metrics:
    coverage_track = []
    total_energy_track = []
    avg_energy_track = []
    for seed in metrics[config]:
        metric = metrics[config][seed]

        # Coverage
        coverage_str = metric['Coverage Rate']
        coverage_track.append(float(coverage_str.strip().replace('%', '')))

        # Total Energy
        total_energy_str = metric['Total Energy Usage']
        total_energy_track.append(float(total_energy_str))

    # Coverage Rate
    coverage_track.sort()
    coverage_track = np.array(coverage_track)

    # Average and Std for coverage
    average_coverage = np.mean(coverage_track)
    std_coverage = np.std(coverage_track)

    # Total energy
    total_energy_track.sort()
    total_energy_track = np.array(total_energy_track)

    # Highest, Lowest, Average, Std for energy
    highest = np.max(total_energy_track)
    lowest = np.min(total_energy_track)
    avg_energy = np.mean(total_energy_track)
    std_energy = np.std(total_energy_track)

    # Print results
    print(f'Reward function [{config}]')
    print(f"{'Highest':<10} {'Lowest':<10} {'Avg Energy':<12} {'Std Energy':<12} {'Avg Coverage':<14} {'Std Coverage':<14}")
    print(f"{highest:<10.2f} {lowest:<10.2f} {avg_energy:<12.2f} {std_energy:<12.2f} {average_coverage:<14.2f} {std_coverage:<14.2f}")

    print('-----------------')
