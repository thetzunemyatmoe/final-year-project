import matplotlib.pyplot as plt
import numpy as np

# Define the parameters
num_episodes = 5000  # Total number of episodes
epsilon_start = 1.0  # Starting value of epsilon

# Different decay rates
decay_rate_1 = 0.9948  # Fast decay
decay_rate_2 = 0.9984  # Medium decay
decay_rate_3 = 0.9995  # Slow decay

# Initialize lists to store epsilon values over episodes
epsilon_values_1 = []
epsilon_values_2 = []
epsilon_values_3 = []

# Calculate epsilon decay over episodes for each decay rate without a minimum limit
epsilon = epsilon_start
for episode in range(num_episodes):
    epsilon_values_1.append(epsilon)
    epsilon *= decay_rate_1

epsilon = epsilon_start
for episode in range(num_episodes):
    epsilon_values_2.append(epsilon)
    epsilon *= decay_rate_2

epsilon = epsilon_start
for episode in range(num_episodes):
    epsilon_values_3.append(epsilon)
    epsilon *= decay_rate_3

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values_1, label=f'Decay rate = Fast', color='blue')
plt.plot(epsilon_values_2, label=f'Decay rate = Medium', color='green')
plt.plot(epsilon_values_3, label=f'Decay rate = Slow', color='orange')

# Add horizontal line at y = 0.01
plt.axhline(y=0.01, color='red', linestyle='--', label='y = 0.01 (Reference Line)')

# Add vertical line at x = 3000
plt.axvline(x=3000, color='purple', linestyle='--', label='x = 3000 (Reference Episode)')

# Labels and title
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay over Episodes for Different Decay Rates')
plt.legend()
plt.grid()
plt.show()
