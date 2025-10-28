# To run, type: python Multiple_ss.py LIST EXPERIMENT DIRECTORY NAMES -id IDENTIFIERS FOR .CSV -o /path/to/output_folder

#Load dependencies
import os
import argparse
import seaborn as sns
sns.__version__
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import pdb  # Import the Python Debugger module

# Set up colorblind-friendly color palette
color_palette = sns.color_palette("colorblind")
home_dir = "/Volumes/holtl02lab/holtl02labspace/Holt_Lab_Members/Nora_Holt/GEM_Experiments"

def concatenate_csv_files(input_dirs, identifiers, identifier_str, output_dir):
    # Check if the number of identifiers matches the number of input files
    if len(input_dirs) != len(identifiers):
        raise ValueError("The number of identifiers must match the number of input files.")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    merged_file = os.path.join(output_dir, f"{identifier_str}_allstepsizes_merged.csv")
    modified_file = os.path.join(output_dir, f"{identifier_str}_allstepsizes_modified.csv")

    # Case 1: Both merged and modified exist -> read modified
    if os.path.exists(merged_file) and os.path.exists(modified_file):
        print(f"Found both merged and modified CSVs. Loading modified: {modified_file}")
        all_data = pd.read_csv(modified_file)
        return all_data

    # Case 2: Only merged exists -> create modified
    if os.path.exists(merged_file):
        print(f"Found merged CSV only. Loading merged file: {merged_file}")
        all_data = pd.read_csv(merged_file)
    else:
        # Case 3: Neither exists or only directories provided -> create merged from raw files
        if len(input_dirs) != len(identifiers):
            raise ValueError("The number of identifiers must match the number of input files.")

        dataframes = []

        for dir in input_dirs:
            try:
                angle_df = pd.read_csv(f"{home_dir}/{dir}/Results/all_data_step_sizes.txt", sep='\t', index_col=0)
                input_values_df = pd.read_csv(f"{home_dir}/{dir}/Results/input_files.csv")
                input_values_df.columns = ["file name", "roi"]

                # Merge to keep only rows matching input values
                angle_df = angle_df.merge(input_values_df, on=["file name", "roi"])
                angle_df = angle_df.replace('', np.NaN)

                dataframes.append(angle_df)
                print(f"Merged files in: {dir}")
            except Exception as e:
                print(f"Could not merge files in {dir}: {e}")

        all_data = pd.concat(dataframes, ignore_index=True)

        # Save merged CSV
        all_data.to_csv(merged_file, index=False)
        print(f"Concatenated CSV saved in: {merged_file}")

    # --- Create modified CSV ---
    all_data = all_data.dropna(subset=['roi'])
    all_data['roi'] = all_data['roi'].fillna('').astype(str)
    if 'directory' not in all_data.columns:
        all_data['directory'] = ''  # Ensure column exists
    all_data['unique_roi'] = all_data['file name'] + '-' + all_data['roi'] + '-' + all_data['directory']

    # Display unique groups for manual merging
    print("Unique groups in the 'group' column:")
    print(all_data['group'].unique())
    print("Entering debugging mode for manual group merging...")
    pdb.set_trace()  # Opens interactive debugging session here

        # Replace group name
        #all_data['group'] = all_data['group'].replace({'WT HSV-1': 'HSV-1','GROUP2':'GROUP2_NEW', etc})

        ### Continue Execution: Type c and press Enter to continue with the rest of the script after making edits.

    # Save modified CSV
    all_data.to_csv(modified_file, index=False)
    print(f"Modified CSV saved in: {modified_file}")

    return all_data

def select_groups(all_data):
    # Print available groups
    unique_groups = all_data['group'].unique().tolist()
    print("Unique groups found:", unique_groups)

    # Always define which_groups before debugger
    which_groups = unique_groups

    # Drop into debugger so you can manually define which_groups
    print("Entering debugging mode for manual group selection: which_groups = ['GroupA', 'GroupB']")
    pdb.set_trace()  
    ### Inside debugger, set groups:
        # which_groups = ['GroupA', 'GroupB']
    #type c + enter to continue

    return which_groups

def ss_distributions(all_data, identifier_str, output_dir):
    ss_csv = os.path.join(output_dir, f"{identifier_str}_stepsize_distributions.csv")

    # Histogram parameters
    min_pts = 0
    max_ss = 1  # microns
    bin_size = 0.02

    # --- Check if CSV exists ---
    if os.path.exists(ss_csv):
        print(f"Found existing step size distributions CSV. Loading: {ss_csv}")
        new_df = pd.read_csv(ss_csv)

        # --- Select groups ---
        selected_groups = select_groups(all_data)

        # Make a copy and filter
        copy_all = all_data.copy()
        copy_all = copy_all[copy_all['group'].isin(selected_groups)]
    else:
        # --- Select groups ---
        selected_groups = select_groups(all_data)

        # Make a copy and filter
        copy_all = all_data.copy()
        copy_all = copy_all[copy_all['group'].isin(selected_groups)]

    # Remove NaNs in 'group'
    copy_all.replace('', np.NaN, inplace=True)
    copy_all.dropna(subset=['group'], inplace=True)

    # Determine step size columns
    start_pos = copy_all.columns.get_loc("0")
    stop_pos = len(copy_all.columns) - start_pos
    step_cols = copy_all.columns[start_pos:start_pos + stop_pos]

    # Create new DataFrame for histogram counts/proportions
    data_arr = []
    for tlag in copy_all['tlag'].unique():
        for bin_left in np.arange(0, max_ss + bin_size, bin_size):
            data_arr.append([tlag, bin_left])
    new_df = pd.DataFrame(data_arr, columns=['tlag', 'bin'])

    # Grouping info
    group_col = 'group'
    my_groups = copy_all[group_col].unique()
    my_tlags = copy_all['tlag'].unique()
    ngroups = len(my_groups)

    # --- Fill histogram / proportion data ---
    for group in my_groups:
        new_df[f"{group}-prop"] = 0
        new_df[f"{group}-count"] = 0
        new_df[f"{group}-total"] = 0

        for tlag in my_tlags:
            # Select all ROIs for this group and tlag
            cur_df = copy_all[(copy_all[group_col] == group) & (copy_all['tlag'] == tlag)]
            obs_dist = pd.to_numeric(cur_df.loc[:, step_cols].to_numpy().flatten(), errors='coerce')
            obs_dist = obs_dist[~np.isnan(obs_dist)]

            n = len(obs_dist)
            new_df.loc[new_df['tlag'] == tlag, f"{group}-total"] = n

            if n > min_pts:
                counts, bins = np.histogram(obs_dist, bins=np.arange(0, max_ss + bin_size, bin_size))
                for i, val in enumerate(counts):
                    new_df.loc[
                        (new_df['tlag'] == tlag) & (new_df['bin'] == bins[i]),
                        f"{group}-prop"
                    ] = val / n
                    new_df.loc[
                        (new_df['tlag'] == tlag) & (new_df['bin'] == bins[i]),
                        f"{group}-count"
                    ] = val

    # Save histogram data CSV
    os.makedirs(output_dir, exist_ok=True)
    new_df.to_csv(ss_csv, index=False)
    print(f"Step size histogram data saved to {ss_csv}")

    # --- Plotting ---
    cmap = plt.get_cmap("viridis")

    group_labels = []
    second_moments = []
    alpha2s = []

    for i, group in enumerate(my_groups):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for tlag in my_tlags:
            cur_df = copy_all[(copy_all[group_col] == group) & (copy_all['tlag'] == tlag)]
            obs_dist = pd.to_numeric(cur_df.loc[:, step_cols].to_numpy().flatten(), errors='coerce')
            obs_dist = obs_dist[~np.isnan(obs_dist)]

            if len(obs_dist) > min_pts:
                alpha = tlag / max(my_tlags)
                color = to_rgba(cmap(i / (ngroups - 1)), alpha=alpha)

                sns.kdeplot(data=obs_dist, ax=ax, label=f'Tlag {tlag}', color=color, alpha=1-tlag/max(my_tlags))

                second_moment = np.mean(obs_dist ** 4) / (3 * (np.mean(obs_dist ** 2) ** 2)) - 1
                alpha2 = second_moment
                second_moments.append(second_moment)
                alpha2s.append(alpha2)
                group_labels.append(group)

        ax.set_title(f'{group}')
        ax.set_ylim(0.001, 10)
        ax.set_xlim(-0.25, 3)
        ax.set_xlabel('Step Size (um)')
        ax.set_yscale('log')
        ax.fill_between(x=[-0.25, 0.0928], y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], color='gray', alpha=0.3)
        ax.legend(fontsize='12', loc='upper right')
        plt.savefig(f"{output_dir}/{identifier_str}_{group}_ss.png", bbox_inches='tight')
        plt.close(fig)

    # --- Plot alpha2 ---
    plt.figure(figsize=(10, 6))
    scatter_width = 0.2

    for i, group in enumerate(my_groups):
        color = cmap(i / (ngroups - 1), 1)
        indices = [idx for idx, label in enumerate(group_labels) if label == group]
        x_coordinates = [i + idx * scatter_width for idx in range(len(indices))]
        plt.scatter(x_coordinates, [alpha2s[idx] for idx in indices], label=f'Group {group}', color=color, s=50)

    plt.xlabel('Group')
    plt.ylabel('Alpha2 Parameter for Each Distribution')
    tick_positions = [i + ((len(my_tlags) - 1) * scatter_width) / 2 for i in range(ngroups)]
    plt.xticks(tick_positions, my_groups, rotation=45)
    plt.savefig(f"{output_dir}/{identifier_str}_alpha2.png", bbox_inches='tight')
    plt.close()
    print(f"Alpha2 plots saved to {output_dir}") 



def main():
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files into one, with unique identifiers, saved in a specific output directory.")
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="List of experiment directory names to concatenate"
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
    
    # Create the output file name based on identifiers
    identifier_str = "_".join(args.identifiers)
    # Concatenate CSV files and save
    all_data = concatenate_csv_files(args.input_dirs, args.identifiers, identifier_str, args.output_dir)
    print(type(all_data))  # This should print <class 'pandas.core.frame.DataFrame'>
    # Generate group comparisons and pass all_data
    ss_distributions(all_data, identifier_str, args.output_dir)

if __name__ == "__main__":
    main()