# To run, type: python Multiple_angles.py LIST EXPERIMENT DIRECTORY NAMES -id IDENTIFIERS FOR .CSV -o /path/to/output_folder

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
    merged_file = os.path.join(output_dir, f"{identifier_str}_allangles_merged.csv")
    modified_file = os.path.join(output_dir, f"{identifier_str}_allangles_modified.csv")

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
                angle_df = pd.read_csv(f"{home_dir}/{dir}/Results/all_data_angles.txt", sep='\t', index_col=0)
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

def angle_distributions(all_data, identifier_str, output_dir):
    # Path to the angle distributions CSV
    angle_csv = os.path.join(output_dir, f"{identifier_str}_angle_distributions.csv")

    # No histogram if the number of step sizes is 9 or less for a given tlag/condition
    min_pts=9
    max_ss=1 # microns
    bin_size=0.05

    # Check if CSV exists
    if os.path.exists(angle_csv):
        print(f"Found existing angle distributions CSV. Loading: {angle_csv}")
        new_df = pd.read_csv(angle_csv)

        # --- Select groups first ---
        selected_groups = select_groups(all_data)

        # Make a copy
        copy_all = all_data.copy()

        # Filter only the selected groups
        copy_all = copy_all[copy_all['group'].isin(selected_groups)]

        print("These are the tlags that are found:\n")
        for tlag in copy_all['tlag'].unique():
            print(tlag)

        # Ensure no NaNs in 'group'
        copy_all = copy_all.replace('', np.NaN)
        copy_all.dropna(axis=0, subset=['group'], inplace=True)

        # Find the last column containing data
        start_pos = copy_all.columns.get_loc("0")
        stop_pos = len(copy_all.columns) - start_pos - 1
        
        group_col='group'
        my_groups=copy_all[group_col].unique()
        my_tlags=copy_all['tlag'].unique()
        #my_tlags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ngroups=len(my_groups)

    else:

        # --- Select groups first ---
        selected_groups = select_groups(all_data)

        # Make a copy
        copy_all = all_data.copy()

        # Filter only the selected groups
        copy_all = copy_all[copy_all['group'].isin(selected_groups)]

        print("These are the tlags that are found:\n")
        for tlag in copy_all['tlag'].unique():
            print(tlag)

        # Ensure no NaNs in 'group'
        copy_all = copy_all.replace('', np.NaN)
        copy_all.dropna(axis=0, subset=['group'], inplace=True)

        # Find the last column containing data
        start_pos = copy_all.columns.get_loc("0")
        stop_pos = len(copy_all.columns) - start_pos - 1

        #set up the new dataframe to save the data
        data_arr=[]
        for tlag in copy_all['tlag'].unique():
            for bin_left in np.arange(0, max_ss + bin_size, bin_size):
                data_arr.append([tlag, bin_left])
        new_df = pd.DataFrame(data_arr, columns=['tlag','bin'])

        # Fill the data frame with the counts per bin for each "group" for the groups and tlags that I want to see
        group_col='group'
        my_groups=copy_all[group_col].unique()
        my_tlags=copy_all['tlag'].unique()
        #my_tlags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ngroups=len(my_groups)

        for group in my_groups:

            new_df[f"{group}-prop"]=0
            new_df[f"{group}-count"]=0
            new_df[f"{group}-total"]=0

            for tlag in my_tlags:

                cur_df = copy_all[(copy_all[group_col]==group) & (copy_all['tlag']==tlag)]
                #print(cur_df.columns[:20])
                #print(type(cur_df.columns[0]))
                # Convert to numeric and flatten, safely ignoring non-numeric entries
                obs_dist = pd.to_numeric(
                    cur_df.iloc[:, start_pos:start_pos+stop_pos].values.flatten(),
                    errors="coerce"
                )
                obs_dist = obs_dist[~np.isnan(obs_dist)]

                new_df.loc[(new_df['tlag']==tlag), f"{group}-total"] = len(obs_dist)

                if(len(obs_dist) > min_pts):
                    n=len(obs_dist)
                    counts, bins = np.histogram(obs_dist, bins=np.arange(0, max_ss + bin_size, bin_size))

                    for i,val in enumerate(counts):
                        new_df.loc[(new_df['tlag']==tlag) & (new_df['bin']==bins[i]), f"{group}-prop"] = val/n
                        new_df.loc[(new_df['tlag']==tlag) & (new_df['bin']==bins[i]), f"{group}-count"] = val

        #save distributions to csv
        new_df.to_csv(angle_csv, index=False)
        print(f"Angle distributions CSV saved: {angle_csv}")

    # matplotlib dictionary of plot parameters
    plt.rcParams['font.size'] = 12

    # Define consistent bins for plotting
    plot_bins = np.arange(0, 181, 5)  # 5-degree bins over 0-180 degrees

    # Define viridis colormap
    cmap = plt.get_cmap("viridis")

    for i, group in enumerate(my_groups):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # Single plot for each group

        # Create a single color for the group
        color = to_rgba(cmap(i / (ngroups - 1))) if ngroups > 1 else to_rgba(cmap(0))

        for tlag in my_tlags:
            cur_df = copy_all[(copy_all[group_col] == group) & (copy_all['tlag'] == tlag)]
            #print(cur_df.columns[:20])
            #print(type(cur_df.columns[0]))
            obs_dist = pd.to_numeric(
                    cur_df.iloc[:, start_pos:start_pos+stop_pos].values.flatten(),
                    errors="coerce"
                )
            obs_dist = obs_dist[~np.isnan(obs_dist)]

            if len(obs_dist) > min_pts:
                # Adjust the alpha scaling for a steeper transition
                # Plot step plot
                plt.hist(obs_dist, bins='auto', alpha=1-tlag/max(my_tlags), label=f'Tlag {tlag}', histtype='step', linewidth=1.5, color=color, density=True)

        # Simulate random behavior (uniform distribution) and plot in gray with dashed lines
        random_data = np.random.uniform(0, 180, size=10000)  # Adjust the size as needed
        plt.hist(random_data, bins=plot_bins, alpha=0.5, histtype='step', linewidth=1.5, linestyle='--', color='gray', density=True, label='Uniform random')

        ax.set_title(f'{group}', fontsize=30)
        ax.set_ylim(0,0.0125)  # Set common y-axis range

        # Move legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        ax.set_xlabel('Angle (Degrees)', fontsize=20)
        ax.set_ylabel('Probability Density', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0, 180)
        plt.savefig(f"{output_dir}/{identifier_str}_{group}_angles.png", bbox_inches='tight')
        plt.close(fig)
        print(f"Angle distribution analysis completed for {group}") 

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
    angle_distributions(all_data, identifier_str, args.output_dir)

if __name__ == "__main__":
    main()
