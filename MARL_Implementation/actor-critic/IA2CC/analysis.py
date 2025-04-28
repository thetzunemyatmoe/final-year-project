import os
import json
import pandas as pd

# Settings
base_dir = 'evaluate'
text_output_file = 'result_table.txt'
csv_output_file = 'result_table.csv'

# Collect data
rows = []

for seed_folder in os.listdir(base_dir):
    seed_path = os.path.join(base_dir, seed_folder)
    if not os.path.isdir(seed_path) or not seed_folder.startswith('seed_'):
        continue

    seed_num = int(seed_folder.split('_')[1])

    for reward_folder in os.listdir(seed_path):
        reward_path = os.path.join(seed_path, reward_folder)
        if not os.path.isdir(reward_path) or not reward_folder.startswith('rewardweight'):
            continue

        reward_num = int(reward_folder.replace('rewardweight', ''))

        stats_file = os.path.join(reward_path, 'config5', 'statistics.json')

        if not os.path.exists(stats_file):
            print(f"Warning: Missing {stats_file}")
            continue

        with open(stats_file, 'r') as f:
            stats = json.load(f)

        try:
            coverage_rate_str = stats['Metic']['Coverage Rate']
            coverage_rate = float(coverage_rate_str.strip(' %'))
            total_energy = stats['Metic']['Total Energy Usage']

            rows.append(
                (seed_num, f"rewardweight{reward_num}",
                 int(coverage_rate), int(total_energy))
            )

        except KeyError as e:
            print(f"Error parsing {stats_file}: missing {e}")
            continue


# Sort the rows
rows.sort()

# Save pretty table (.txt)
col_widths = [10, 20, 20, 15]
header = f"{'Seed'.ljust(col_widths[0])}|{'Reward Weight'.ljust(col_widths[1])}|{'Coverage Rate'.ljust(col_widths[2])}|{'Total Energy'.ljust(col_widths[3])}"
separator = '-' * (sum(col_widths) + 3)

lines = [header, separator]

current_seed = None

for seed, reward, coverage, energy in rows:
    if current_seed is None:
        current_seed = seed
    elif seed != current_seed:
        lines.append('')  # Blank line between different seeds
        current_seed = seed

    line = f"{str(seed).ljust(col_widths[0])}|{reward.ljust(col_widths[1])}|{f'{coverage:.2f}'.ljust(col_widths[2])}|{str(energy).ljust(col_widths[3])}"
    lines.append(line)

with open(text_output_file, 'w') as f:
    for line in lines:
        f.write(line + '\n')

print(f"\n✅ Text table saved successfully to '{text_output_file}'!")

# Save as CSV for easy analysis
df = pd.DataFrame(
    rows, columns=["Seed", "Reward Weight", "Coverage Rate", "Total Energy"])
df.to_csv(csv_output_file, index=False)

print(f"✅ CSV file saved successfully to '{csv_output_file}'!")

# Optional: show quick stats
print("\n📊 Quick Summary (Coverage Rate per Reward Weight):")
print(df.groupby('Reward Weight')["Coverage Rate"].mean())
