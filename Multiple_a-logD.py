# To run, type: python Multiple_a-logD.py LIST ALL-DATA_FILTERED.CSV FILEPATHS -id IDENTIFIERS FOR .CSV -o /path/to/output_folder

#Load dependencies
import os
import argparse
import seaborn as sns
sns.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import pdb  # Import the Python Debugger module

# Set up colorblind-friendly color palette
color_palette = sns.color_palette("colorblind")

def concatenate_csv_files(input_files, identifiers, output_dir):
    # Check if the number of identifiers matches the number of input files
    if len(input_files) != len(identifiers):
        raise ValueError("The number of identifiers must match the number of input files.")
    
    # List to store individual dataframes
    dataframes = []
    
    # Read each CSV file and append to the list
    for file in input_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Loaded file: {file}")
        except Exception as e:
            print(f"Could not read file {file}: {e}")
    
    # Concatenate all dataframes
    all_data = pd.concat(dataframes, ignore_index=True)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the output file name based on identifiers
    identifier_str = "_".join(identifiers)
    output_file = os.path.join(output_dir, f"{identifier_str}_alldata_merged.csv")
    
    # Save concatenated DataFrame to a CSV file
    all_data.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved in: {output_file}")

    # Drop rows with missing values in 'roi'
    all_data = all_data.dropna(subset=['roi'])
    
    # Fill missing values in 'roi' column with an empty string
    all_data['roi'] = all_data['roi'].fillna('')
    
    # Convert the 'roi' column to string (str) data type
    all_data['roi'] = all_data['roi'].astype(str)
    
    # Create the 'unique_roi' column
    all_data['unique_roi'] = all_data['file name'] + '-' + all_data['roi'] + '-' + all_data['directory']
    
    # Display unique groups and pause for manual merging
    print("Unique groups in the 'group' column:")
    print(all_data['group'].unique())

    # Set a breakpoint for manual group merging
    print("Entering debugging mode for manual group merging...")
    pdb.set_trace()  # Opens interactive debugging session here

        # Replace group name
        #all_data['group'] = all_data['group'].replace({'WT HSV-1': 'HSV-1','GROUP2':'GROUP2_NEW', etc})

        ### Continue Execution: Type c and press Enter to continue with the rest of the script after making edits.

    # Save the modified DataFrame after debugging
    modified_output = os.path.join(output_dir, f"{identifier_str}_alldata_modified.csv")
    all_data.to_csv(modified_output, index=False)
    print(f"Concatenated CSV saved in: {modified_output}")

    return all_data

# Function to compute group statistics
def compute_group_stats(data, group):
    group_data = data[data['group'] == group].dropna(subset=['aexp'])
    medians = {
        'alpha': np.nanmedian(group_data['aexp']),
        'logD': np.nanmedian(group_data['log_D']),
        'D': np.nanmedian(group_data['D'])
    }
    centroids = group_data.groupby('unique_roi').agg({'log_D': 'mean', 'aexp': 'mean'})
    return group_data, medians, centroids

# Function to create combined scatter plot and histograms
def create_combined_plot(group1_data, group2_data, medians1, medians2, centroids1, centroids2,
                         group1, group2, output_dir, identifier_str, all_data,
                         group1_color='#666666', group2_color='#ff0066'): #group1 was original blue, group2 was yellow/orange
    fig, ax = plt.subplots(figsize=(6, 6))

    # -- Lighten the color for small foci scatterplots
    light_group1 = mcolors.to_rgba(group1_color, alpha=0.3)
    light_group2 = mcolors.to_rgba(group2_color, alpha=0.3)

    # Plotting data and KDEs—Small points (foci), can choose one or both
    sns.scatterplot(data=group1_data, x='log_D', y='aexp', s=2, ax=ax, color=light_group1, label=group1, alpha=0.6)
    sns.scatterplot(data=group2_data, x='log_D', y='aexp', s=2, ax=ax, color=light_group2, label=group2, alpha=0.6)
    sns.kdeplot(data=group1_data, x='log_D', y='aexp', ax=ax, cmap=sns.light_palette(group1_color, as_cmap=True), levels=10, alpha=0.3, linewidths=2)
    sns.kdeplot(data=group2_data, x='log_D', y='aexp', ax=ax, cmap=sns.light_palette(group2_color, as_cmap=True), levels=10, alpha=0.3, linewidths=2)

    # Plotting centroids
    ax.scatter(centroids1['log_D'], centroids1['aexp'], s=50, color=group1_color, edgecolor='black', marker='o')
    ax.scatter(centroids2['log_D'], centroids2['aexp'], s=50, color=group2_color, edgecolor='black', marker='o')

    # Configure plot
    ax.set_xlim(-3, 2)
    ax.set_ylim(-1, 2)
    ax.set_xlabel('$log(D_{100ms})$', fontsize=24)
    ax.set_ylabel('α', fontsize=30)

    # Histograms along the X-axis
    counts_x_group1, bins_x_group1 = np.histogram(group1_data['log_D'], bins=np.linspace(-3, 2, 31))
    counts_x_group2, bins_x_group2 = np.histogram(group2_data['log_D'], bins=np.linspace(-3, 2, 31))
    rel_freq_x_group1 = counts_x_group1 / float(len(group1_data))
    rel_freq_x_group2 = counts_x_group2 / float(len(group2_data))

    ax_hist_x = fig.add_axes([0.1, 0.95, 0.8, 0.08])
    ax_hist_x.bar(bins_x_group1[:-1], rel_freq_x_group1, width=np.diff(bins_x_group1),
                  color=group1_color, align='edge', alpha=0.5, label=group1)
    ax_hist_x.bar(bins_x_group2[:-1], rel_freq_x_group2, width=np.diff(bins_x_group2),
                  color=group2_color, align='edge', alpha=0.5, label=group2)
    ax_hist_x.set_xlim(-3, 2)
    ax_hist_x.yaxis.set_tick_params(labelleft=False)

    # Histograms along the Y-axis
    counts_y_group1, bins_y_group1 = np.histogram(group1_data['aexp'], bins=np.linspace(-1, 2, 31))
    counts_y_group2, bins_y_group2 = np.histogram(group2_data['aexp'], bins=np.linspace(-1, 2, 31))
    rel_freq_y_group1 = counts_y_group1 / float(len(group1_data))
    rel_freq_y_group2 = counts_y_group2 / float(len(group2_data))

    ax_hist_y = fig.add_axes([0.98, 0.1, 0.08, 0.8])
    ax_hist_y.barh(bins_y_group1[:-1], rel_freq_y_group1, height=np.diff(bins_y_group1),
                   color=group1_color, align='edge', alpha=0.5, label=group1)
    ax_hist_y.barh(bins_y_group2[:-1], rel_freq_y_group2, height=np.diff(bins_y_group2),
                   color=group2_color, align='edge', alpha=0.5, label=group2)
    ax_hist_y.set_ylim(-1, 2)
    ax_hist_y.xaxis.set_tick_params(labelbottom=False)

    # Plot horizontal and vertical lines at medians
    ax.axhline(y=medians1['alpha'], color=group1_color, linestyle='--', linewidth=3)
    ax.axvline(x=medians1['logD'], color=group1_color, linestyle='--', linewidth=3)
    ax.axhline(y=medians2['alpha'], color=group2_color, linestyle='--', linewidth=3)
    ax.axvline(x=medians2['logD'], color=group2_color, linestyle='--', linewidth=3)

    # White background boxes for median text annotations
    def draw_text_box(ax, x, y, text, color, rotation=0):
        box = patches.FancyBboxPatch(
            (x - 0.1, y - 0.05),  # Adjust to fit box behind text
            width=0.8, height=0.15,
            boxstyle="round,pad=0.05",
            color="white", zorder=1, transform=ax.transData
        )
        ax.add_patch(box)
        ax.text(x, y, text, color=color, fontsize=18, zorder=2, rotation=rotation)

    # Replacing text calls with boxes + text
    draw_text_box(ax, -2.5, 1.2, f"$α={medians1['alpha']:.2f}$", group1_color)
    draw_text_box(ax, 1, -0.8, f"$D={medians1['D']:.2f}$", group1_color, rotation=270)
    draw_text_box(ax, -2.5, 0.8, f"$α={medians2['alpha']:.2f}$", group2_color)
    draw_text_box(ax, 1.6, -0.8, f"$D={medians2['D']:.2f}$", group2_color, rotation=270)
    
    # Display medians on plot
    ax.text(-2.5, 1.2, f"$α={medians1['alpha']:.2f}$", color=group1_color, fontsize=18)
    ax.text(1, -0.8, f"$D={medians1['D']:.2f}$", rotation=270, color=group1_color, fontsize=18)
    ax.text(-2.5, 0.8, f"$α={medians2['alpha']:.2f}$", color=group2_color, fontsize=18)
    ax.text(1.6, -0.8, f"$D={medians2['D']:.2f}$", rotation=270, color=group2_color, fontsize=18)

    # Save plot
    plt.savefig(f'{output_dir}/{identifier_str}_{group1}-vs-{group2}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot for {group1} vs. {group2} saved")
    
    # **Add Statistical Analysis Below**
    # Calculate centroids for each group
    centroids_group1 = group1_data.groupby('unique_roi').agg({'D': 'median', 'aexp': 'median'})
    centroids_group2 = group2_data.groupby('unique_roi').agg({'D': 'median', 'aexp': 'median'})

    # Calculate median values for each directory and save
    def median_values(df):
        return df.groupby('group').agg({'D': 'median', 'aexp': 'median'})
    
    directory_group_medians = all_data.groupby('directory').apply(median_values)
    directory_group_medians.to_csv(f"{output_dir}/{identifier_str}_median_values.csv", index=False)

    # Rename columns for merging
    centroids_group1 = centroids_group1.rename(
        columns={'D': f'{group1}_D_median', 'aexp': f'{group1}_aexp_median'}
    )
    centroids_group2 = centroids_group2.rename(
        columns={'D': f'{group2}_D_median', 'aexp': f'{group2}_aexp_median'}
    )

    # Merge DataFrames on 'unique_roi' and save
    merged_df = pd.merge(
        centroids_group1, centroids_group2, left_index=True, right_index=True, how='outer'
    )
    merged_df.to_csv(f"{output_dir}/{identifier_str}_centroids_{group1}-{group2}.csv", index=False)

    # Perform one-way ANOVA on 'log_D' across groups and save
    model = smf.ols('log_D ~ C(group)', data=all_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(f"{output_dir}/{identifier_str}_anova.csv", index=False)

    # Perform Tukey's HSD test and save results
    tukey_result = pairwise_tukeyhsd(all_data['log_D'], all_data['group'], alpha=0.05/3)
    with open(f"{output_dir}/{identifier_str}_tukey.txt", "w") as file:
        file.write(str(tukey_result))

# Function to run pairwise comparisons
def generate_group_comparisons(all_data, output_dir, identifier_str):
    # Ensure that data is a pandas DataFrame
    if not isinstance(all_data, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame but got {}".format(type(all_data)))

    unique_groups = all_data['group'].unique()
    for group1, group2 in combinations(unique_groups, 2):
        # Compute statistics and centroids for each group
        group1_data, medians1, centroids1 = compute_group_stats(all_data, group1)
        group2_data, medians2, centroids2 = compute_group_stats(all_data, group2)

        # Create and save plot for the current comparison
        create_combined_plot(group1_data, group2_data, medians1, medians2, centroids1, centroids2,
                             group1, group2, output_dir, identifier_str, all_data)


def main():
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files into one, with unique identifiers, saved in a specific output directory.")
    parser.add_argument(
        "input_files",
        nargs="+",
        help="List of CSV files to concatenate"
    )
    parser.add_argument(
        "-id", "--identifiers",
        nargs="+",
        required=True,
        help="List of identifiers corresponding to each CSV file (must match number of input files)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory to save the concatenated CSV and other downstream results"
    )
    args = parser.parse_args()
    
    # Concatenate CSV files and save
    all_data = concatenate_csv_files(args.input_files, args.identifiers, args.output_dir)
    print(type(all_data))  # This should print <class 'pandas.core.frame.DataFrame'>
    # Generate group comparisons and pass all_data
    generate_group_comparisons(all_data, args.output_dir, "_".join(args.identifiers))


if __name__ == "__main__":
    main()

