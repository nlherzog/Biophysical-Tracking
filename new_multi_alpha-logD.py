# To run, type: python new_multi_alpha-logD.py -id IDENTIFIER_STR -o /path/to/output_folder

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
import matplotlib.colors as mcolors

def process_merged_file(identifier_str, output_dir):
    merged_file = os.path.join(output_dir, f"{identifier_str}_alldata_merged.csv")
    modified_output = os.path.join(output_dir, f"{identifier_str}_alldata_modified.csv")
    comparison_file = os.path.join(output_dir, f"{identifier_str}_group_comparisons.csv")
    
    group_comparisons = []

    if os.path.exists(modified_output):
        all_data = pd.read_csv(modified_output)

        if os.path.exists(comparison_file):
            df_comparisons = pd.read_csv(comparison_file)
            group_comparisons = list(df_comparisons.itertuples(index=False, name=None))
            print("Loaded modified data and group comparisons.")
            return all_data, group_comparisons
        else:
            print("\nMissing required file: group_comparisons")
            print(all_data['group'].unique())
            print("    group_comparisons = [(\"GroupA\", \"GroupB\", \"#1f77b4\", \"#ff7f0e\"), ...]")
            pdb.set_trace()
            ### all_data['group'].unique() #USE TO CHECK
            ### Continue Execution: Type c and press Enter to continue with the rest of the script after making edits.
            # Save (unchanged) modified file anyway
            all_data.to_csv(modified_output, index=False)
            print(f"Saved modified output to: {modified_output}")
            return all_data, group_comparisons
        
    else:
        print("\nMissing required file: modified_output")
        if not os.path.exists(merged_file):
            raise FileNotFoundError(f"Merged file not found: {merged_file}")
        
        all_data = pd.read_csv(merged_file)
        print(f"Loaded merged file: {merged_file}")

        # Preprocessing
        all_data = all_data.dropna(subset=['roi'])
        all_data['roi'] = all_data['roi'].fillna('').astype(str)
        all_data['unique_roi'] = all_data['file name'] + '-' + all_data['roi'] + '-' + all_data['directory']

        unique_groups = all_data['group'].unique()
        print("Unique groups in the 'group' column:")
        for idx, group in enumerate(unique_groups):
            print(f"[{idx}] {group}")

        print("\nModified output file was missing.")
        print("Entering debugging mode to optionally edit groups and define comparisons/colors...\n")
        print("Example:")
        print("    all_data['group'] = all_data['group'].replace({'Long name': 'Short'})")
        print("    group_comparisons = [(\"GroupA\", \"GroupB\", \"#1f77b4\", \"#ff7f0e\"), ...]")
        pdb.set_trace()
        '''all_data['group'] = all_data['group'].replace({
            'Long complicated name 1': 'Short1',
            'Long complicated name 2': 'Short2',
            'Another long name': 'Short3' 
        })
            
        group_comparisons = [
            ("Control", "Treated", "#1f77b4", "#ff7f0e"),
            ("Early", "Late", "#2ca02c", "#d62728")
        ]   '''
        ### all_data['group'].unique() #USE TO CHECK
        ### Continue Execution: Type c and press Enter to continue with the rest of the script after making edits.        

        # Save modified version
        all_data.to_csv(modified_output, index=False)
        print(f"Saved modified output to: {modified_output}")

        return all_data, group_comparisons

def darken_color(color, amount=0.7):
    """
    Darkens the given color by multiplying (1 - amount) to RGB channels.
    `amount` < 1 results in a darker color.
    """
    c = mcolors.to_rgb(color)
    return tuple(max(0, min(1, channel * amount)) for channel in c)

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
                         group1_color='#fc7bbc', group2_color='#ff0066'): #group1 was original blue, group2 was yellow/orange
    fig, ax = plt.subplots(figsize=(6, 6))

    dark_group1 = darken_color(group1_color, amount=0.85)
    dark_group2 = darken_color(group2_color, amount=0.85)

    # Plotting data and KDEs
    sns.kdeplot(data=group1_data, x='log_D', y='aexp', ax=ax, cmap=sns.light_palette(dark_group1, as_cmap=True), levels=10, alpha=0.3, linewidths=2)
    sns.kdeplot(data=group2_data, x='log_D', y='aexp', ax=ax, cmap=sns.light_palette(dark_group2, as_cmap=True), levels=10, alpha=0.3, linewidths=2)

    # Plotting centroids
    #ax.scatter(centroids1['log_D'], centroids1['aexp'], s=50, color=group1_color, edgecolor='black', marker='o')
    # Group 1 as open circle (facecolors='none')
    ax.scatter(centroids1['log_D'], centroids1['aexp'], s=50, edgecolor=group1_color,
               facecolors='none', marker='o', linewidth=1.5)  # ← MODIFIED
    ax.scatter(centroids2['log_D'], centroids2['aexp'], s=50, color=group2_color, edgecolor='black', marker='o')

    # Configure plot limits: MANUAL
    ax.set_xlim(-3, 2)
    ax.set_ylim(0, 2)

    """ # Compute dynamic x-limits based on combined log_D values
    all_logD = pd.concat([group1_data['log_D'], group2_data['log_D']])
    x_min = all_logD.min()
    x_max = all_logD.max()
    x_range = x_max - x_min
    x_pad = x_range * 0.1  # 10% padding on both sides
    # Set new x-limits
    ax.set_xlim(x_min - x_pad, x_max + x_pad) """

    # Configure plot
    ax.set_xlabel('$log(D_{100ms})$', fontsize=24)
    ax.set_ylabel('α', fontsize=30)

    # KDEs (density plots) along the X-axis
    ax_hist_x = fig.add_axes([0.1, 0.95, 0.8, 0.08], sharex=ax)
    sns.kdeplot(group1_data['log_D'], ax=ax_hist_x, color=group1_color, linewidth=2)
    sns.kdeplot(group2_data['log_D'], ax=ax_hist_x, color=group2_color, linewidth=2)
    ax_hist_x.set_ylabel('')
    ax_hist_x.set_xlabel('')  # suppress x-label
    ax_hist_x.set_yticks([])
    ax_hist_x.tick_params(axis='x', labelbottom=True)
    ax_hist_x.set_xticks([-3, -2, -1, 0, 1, 2]) #CHANGE SCRIPT HERE FOR MANUAL LIMITS
    ax_hist_x.set_xlim(-3, 2) #CHANGE SCRIPT HERE FOR MANUAL LIMITS
    #x_ticks = np.round(np.linspace(x_min - x_pad, x_max + x_pad, num=6), 1) #CHANGE SCRIPT HERE FOR AUTO LIMITS
    #ax_hist_x.set_xticks(x_ticks) #CHANGE SCRIPT HERE FOR AUTO LIMITS
    #ax_hist_x.set_xlim(x_min - x_pad, x_max + x_pad) #CHANGE SCRIPT HERE FOR AUTO LIMITS

    # KDEs (density plots) along the Y-axis
    ax_hist_y = fig.add_axes([0.98, 0.1, 0.08, 0.8], sharey=ax)
    sns.kdeplot(y=group1_data['aexp'], ax=ax_hist_y, color=group1_color, linewidth=2)
    sns.kdeplot(y=group2_data['aexp'], ax=ax_hist_y, color=group2_color, linewidth=2)
    ax_hist_y.set_xlabel('')
    ax_hist_y.set_ylabel('')  # suppress y-label
    ax_hist_y.set_xticks([])
    ax_hist_y.tick_params(axis='y', labelleft=True)
    ax_hist_y.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax_hist_y.set_ylim(0, 2)

    # Plot horizontal and vertical lines at medians
    ax.axhline(y=medians1['alpha'], color=group1_color, linestyle='--', linewidth=3)
    ax.axvline(x=medians1['logD'], color=group1_color, linestyle='--', linewidth=3)
    ax.axhline(y=medians2['alpha'], color=group2_color, linestyle='--', linewidth=3)
    ax.axvline(x=medians2['logD'], color=group2_color, linestyle='--', linewidth=3)
    
    # Display medians on plot WITH MANUAL AXES
    ax.text(-2.5, 1.7, f"$α={medians1['alpha']:.2f}$", color=group1_color, fontsize=18)
    ax.text(1, 0.1, f"$D={medians1['D']:.3f}$", rotation=270, color=group1_color, fontsize=18)
    ax.text(-2.5, 1.9, f"$α={medians2['alpha']:.2f}$", color=group2_color, fontsize=18)
    ax.text(1.6, 0.1, f"$D={medians2['D']:.3f}$", rotation=270, color=group2_color, fontsize=18)

    """ # Display medians on plot WITH AUTO AXES
    ax.text(x_min + 0.05 * x_range, 1.7, f"$α={medians1['alpha']:.2f}$", color=group1_color, fontsize=18)
    ax.text(x_max - 0.1 * x_range, 0.1, f"$D={medians1['D']:.2f}$", rotation=270, color=group1_color, fontsize=18)
    ax.text(x_min + 0.05 * x_range, 1.9, f"$α={medians2['alpha']:.2f}$", color=group2_color, fontsize=18)
    ax.text(x_max - 0.05 * x_range, 0.1, f"$D={medians2['D']:.2f}$", rotation=270, color=group2_color, fontsize=18) """

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
def generate_group_comparisons(all_data, output_dir, identifier_str, group_comparisons=None):
    if not isinstance(all_data, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame but got {}".format(type(all_data)))

    if group_comparisons is None:
        group_comparisons = list(combinations(all_data['group'].unique(), 2))
        # Default color assignment if not provided
        group_comparisons = [(g1, g2, '#1f77b4', '#ff7f0e') for g1, g2 in group_comparisons]
    
    # Save group_comparisons to CSV if it's not None
    if group_comparisons:
        group_comparisons_df = pd.DataFrame(group_comparisons, columns=['group1', 'group2', 'group1_color', 'group2_color'])
        comparisons_path = os.path.join(output_dir, f"{identifier_str}_group_comparisons.csv")
        group_comparisons_df.to_csv(comparisons_path, index=False)

    for group1, group2, group1_color, group2_color in group_comparisons:
        group1_data, medians1, centroids1 = compute_group_stats(all_data, group1)
        group2_data, medians2, centroids2 = compute_group_stats(all_data, group2)

        create_combined_plot(group1_data, group2_data, medians1, medians2, centroids1, centroids2,
                             group1, group2, output_dir, identifier_str, all_data,
                             group1_color=group1_color, group2_color=group2_color)


def main():
    parser = argparse.ArgumentParser(description="Process a pre-existing merged CSV file and define comparisons.")
    parser.add_argument(
        "-id", "--identifier",
        required=True,
        help="Identifier string used to locate the merged file"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory containing merged file and where outputs will be saved"
    )
    args = parser.parse_args()
    
    # Concatenate CSV files and save
    all_data, group_comparisons = process_merged_file(args.identifier, args.output_dir)

    print(type(all_data))  # This should print <class 'pandas.core.frame.DataFrame'>
    try:
        group_comparisons  # Check if it's defined in pdb
    except NameError:
        group_comparisons = None
    
    # Generate group comparisons and pass all_data
    generate_group_comparisons(all_data, args.output_dir, args.identifier, group_comparisons)

    


if __name__ == "__main__":
    main()

