import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.stats import ks_2samp

# Manually select two groups
group1 = 'NHDF_HSV-1'
group2 = 'NHDF-ICP4_None'

# Initialize lists to store test statistics and p-values for each "tau"
statistics = []
p_values = []

# Extract data for the selected groups
data_group1 = df[df[group_col] == group1]
data_group2 = df[df[group_col] == group2]

# Loop over all "taus"
for tlag in my_tlags:
    # Extract angle distributions for the current "tau"
    obs_dist_group1 = np.asarray(data_group1[data_group1['tlag'] == tlag].loc[:, "0":str(stop_pos)]).flatten()
    obs_dist_group2 = np.asarray(data_group2[data_group2['tlag'] == tlag].loc[:, "0":str(stop_pos)]).flatten()

    # Remove NaN values
    obs_dist_group1 = obs_dist_group1[~np.isnan(obs_dist_group1)]
    obs_dist_group2 = obs_dist_group2[~np.isnan(obs_dist_group2)]

    # Perform Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(obs_dist_group1, obs_dist_group2)

    # Store test results
    statistics.append(statistic)
    p_values.append(p_value)

# Create a ListedColormap with discrete colors for each "tau" value
colors = plt.cm.Greys(np.linspace(0, 1, 10))
# Reverse the order of colors to invert the colormap
colors = colors[::-1]
cmap = ListedColormap(colors)

# Plot the volcano plot with varying transparency and add a color bar
plt.figure(figsize=(8, 6))

# Define transparency based on tau values
max_tau = max(my_tlags)
tau_transparency = [1 - (tau / max_tau) for tau in my_tlags]

# Scatter plot with varying transparency and discrete colors
scatter = plt.scatter(statistics, -np.log10(p_values), c=my_tlags, cmap=cmap, alpha=tau_transparency, vmin=min(my_tlags), vmax=max(my_tlags), s=100)  # Increase dot size here (e.g., s=100)

plt.title('Volcano Plot of Kolmogorov-Smirnov Test')
plt.xlabel('Test Statistic')
plt.ylabel('-log10(p-value)')
plt.grid(False)  # Turn off the grid
plt.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1)  # Add significance threshold line

# Add color bar with discrete values
cbar = plt.colorbar(scatter, ticks=my_tlags)
cbar.set_label('Tau (lag between displacements)', fontsize=12)

plt.show()