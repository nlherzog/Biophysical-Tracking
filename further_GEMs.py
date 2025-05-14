# to call script use terminal below
# python further_GEMs.py -i /PATH-TO-EXPERIMENT-HOME-DIRECTORY

#import dependencies
import multiprocessing
import re
import os
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import to_rgba

#Define functions

#angle distributions (Function 1)
def angle_distributions(home_dir):
    print("Angle distribution analysis started")
    data_dir = f'{home_dir}/Results_250ms'

    # load the angle distribution file
    raw_df1 = pd.read_csv(f"{data_dir}/all_data_angles.txt", sep='\t', index_col=0)
    df = raw_df1
    
    # Read the input values from the CSV file
    input_values_df = pd.read_csv(f"{data_dir}/input_files.csv")

    # Assuming the CSV file has columns "File name" and "ROI"
    input_values_df.columns = ["file name", "roi"]

    # Filter the DataFrame based on matching values in 'file name' and 'roi'
    df = df.merge(input_values_df, on=["file name", "roi"])
    #df now contains only the rows where both "file name" and "roi" match the input values

    # Remove blanks in the table, find the last column containing data
    df = df.replace('', np.NaN)
    df.dropna(axis=0, subset=['group'], inplace=True)
    start_pos = df.columns.get_loc("0")
    stop_pos = len(df.columns) - start_pos - 1

    # No histogram if the number of step sizes is 9 or less for a given tlag/condition
    min_pts=9
    max_ss=1 # microns
    bin_size=0.05

    #set up the new dataframe to save the data
    data_arr=[]
    for tlag in df['tlag'].unique():
        for bin_left in np.arange(0, max_ss + bin_size, bin_size):
            data_arr.append([tlag, bin_left])
    new_df = pd.DataFrame(data_arr, columns=['tlag','bin'])

    # Fill the data frame with the counts per bin for each "group"
    group_col='group'

    # the groups and tlags that I want to see
    my_groups=df[group_col].unique()
    my_tlags=df['tlag'].unique()
    #my_tlags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ngroups=len(my_groups)

    for group in my_groups:

        new_df[f"{group}-prop"]=0
        new_df[f"{group}-count"]=0
        new_df[f"{group}-total"]=0

        for tlag in my_tlags:

            cur_df = df[(df[group_col]==group) & (df['tlag']==tlag)]
            obs_dist = np.asarray(cur_df.loc[:, "0":str(stop_pos-1)]).flatten()
            #obs_dist = np.asarray(cur_df.loc[:, "0":"1"]).flatten()
            obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]

            new_df.loc[(new_df['tlag']==tlag), f"{group}-total"] = len(obs_dist)
            if(len(obs_dist) > min_pts):

                n=len(obs_dist)
                counts, bins = np.histogram(obs_dist, bins=np.arange(0, max_ss + bin_size, bin_size))

                for i,val in enumerate(counts):
                    new_df.loc[(new_df['tlag']==tlag) & (new_df['bin']==bins[i]), f"{group}-prop"] = val/n
                    new_df.loc[(new_df['tlag']==tlag) & (new_df['bin']==bins[i]), f"{group}-count"] = val

    # Plotting the distribution of the step sizes
    #make a directory to save the plots
    plot_dir = os.path.join(home_dir, "Further_Analyses")
    os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist

    #save distributions to csv
    new_df.to_csv(f"{plot_dir}/angle_distributions.csv", index=False)

    # matplotlib dictionary of plot parameters
    plt.rcParams['font.size'] = 12

    # Define the 'viridis' color map
    cmap = plt.get_cmap("viridis")

    for i, group in enumerate(my_groups):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # Single plot for each group

        # Create a single color for the group
        color = to_rgba(cmap(i / (ngroups - 1)))

        for tlag in my_tlags:
            cur_df = df[(df[group_col] == group) & (df['tlag'] == tlag)]
            obs_dist = np.asarray(cur_df.loc[:, "0":str(stop_pos)]).flatten()
            obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]

            if len(obs_dist) > min_pts:
                # Adjust the alpha scaling for a steeper transition
                # Plot step plot
                plt.hist(obs_dist, bins='auto', alpha=1-tlag/max(my_tlags), label=f'Tlag {tlag}', histtype='step', linewidth=1.5, color=color, density=True)

                # Calculate the estimator of the second moment (<x^2>)

        # Simulate random behavior (uniform distribution) and plot in gray with dashed lines
        random_data = np.random.uniform(0, 180, size=10000)  # Adjust the size as needed
        plt.hist(random_data, bins='auto', alpha=0.5, histtype='step', linewidth=1.5, linestyle='--', color='gray', density=True)

        ax.set_title(f'{group}', fontsize=30)
        #ax.set_ylim(0,0.0125)  # Set common y-axis range

        # Move legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=21)
        #plt.xlim(0,180)
        plt.xticks(fontsize=18)
        plt.xlabel('Angle (Degrees)', fontsize=30)
        plt.ylabel('Probability Density', fontsize=30)
        plt.yticks(fontsize=18)
        
        plt.savefig(f"{plot_dir}/{group}_angles.png", bbox_inches='tight')
        plt.close(fig)
    print("Angle distribution analysis completed")

#step size distribution (Function 2)
def ss_distributions(home_dir):
    print("Step size distribution analysis started")
    data_dir = f'{home_dir}/Results_250ms'

    # load the step size distribution file
    raw_df = pd.read_csv(f"{data_dir}/all_data_step_sizes.txt", sep='\t', index_col=0)
    df = raw_df
    
    # Read the input values from the CSV file
    input_values_df = pd.read_csv(f"{data_dir}/input_files.csv")

    # Assuming the CSV file has columns "File name" and "ROI"
    input_values_df.columns = ["file name", "roi"]

    # Filter the DataFrame based on matching values in 'file name' and 'roi'
    df = df.merge(input_values_df, on=["file name", "roi"])
    #df now contains only the rows where both "file name" and "roi" match the input values

    # Remove blanks in the table, find the last column containing data
    df = df.replace('', np.NaN)
    df.dropna(axis=0, subset=['group'], inplace=True)
    df.dropna(axis=0, subset=['roi'], inplace=True)
    start_pos = df.columns.get_loc("0")
    stop_pos = len(df.columns) - start_pos - 1  

    # Set up a new data frame to save the data
    # No histogram if the number of step sizes is 9 or less for a given tlag/condition
    min_pts = 9
    max_ss = 1  # microns
    bin_size = 0.02

    data_arr = []
    for tlag in df['tlag'].unique():
        for bin_left in np.arange(0, max_ss + bin_size, bin_size):
            data_arr.append([tlag, bin_left])
    new_df = pd.DataFrame(data_arr, columns=['tlag', 'bin'])

    """     # Fill the data frame with the counts per bin for each "roi" MAY NEED TO DELETE THIS WHOLE PART: POSSIBLY UNNECESSARY
    group_col = 'roi'

    my_groups = df[group_col].unique()
    my_tlags = df['tlag'].unique()

    for group in my_groups:

        new_df[f"{group}-prop"]=0
        new_df[f"{group}-count"]=0
        new_df[f"{group}-total"]=0

        for j, tlag in enumerate(my_tlags):

            cur_df = df[(df[group_col]==group) & (df['tlag']==tlag)]
            obs_dist = np.asarray(cur_df.loc[:, "0":str(stop_pos)]).flatten()
            obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]

            n = len(obs_dist)
            new_df.loc[(new_df['tlag'] == tlag), f"{group}-total"] = n

            if(len(obs_dist) > min_pts):
                counts, bins = np.histogram(obs_dist, bins=np.arange(0, max_ss + bin_size, bin_size))

                for i, val in enumerate(counts):
                    # Get the 'group' value for the current 'group' and add it to the column name
                    group_value = df[df['roi'] == group]['group'].values[0]
                    new_df.loc[(new_df['tlag'] == tlag) & (new_df['bin'] == bins[i]), f"{group}-{group_value}-prop"] = val / n
                    new_df.loc[(new_df['tlag'] == tlag) & (new_df['bin'] == bins[i]), f"{group}-{group_value}-count"] = val """

    # Define the 'viridis' color map
    cmap = plt.get_cmap("viridis")

    # Plotting - modified to create a single plot for each group with all taus
    group_col = 'group'
    
    #make a directory to save the plots
    plot_dir = os.path.join(home_dir, "Further_Analyses")
    os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # the groups and tlags that I want to see
    my_groups = df[group_col].unique()
    my_tlags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ngroups = len(my_groups)

    group_labels = []  # List to store group labels for x-axis
    second_moments = []  # List to store the estimator of the second moment for each distribution
    alpha2s = []

    for i, group in enumerate(my_groups):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # Single plot for each group
        for tlag in my_tlags:
            cur_df = df[(df[group_col] == group) & (df['tlag'] == tlag)]
            obs_dist = np.asarray(cur_df.loc[:, "0":str(stop_pos)]).flatten()
            obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]

            if len(obs_dist) > min_pts:
                # Adjust the alpha scaling for a steeper transition
                alpha = tlag / max(my_tlags)  # Square the factor for a steeper transition

                # Convert color to RGBA and set alpha value
                color = to_rgba(cmap(i / (ngroups - 1)), alpha=alpha)

                sns.kdeplot(data=obs_dist, ax=ax, label=f'Tlag {tlag}', color=color,alpha=1-tlag/max(my_tlags))

                # Calculate the estimator of the second moment (<x^2>)
                second_moment = np.mean(obs_dist ** 4) / (3 * (np.mean(obs_dist ** 2) ** 2)) - 1
                alpha2 = np.mean(obs_dist ** 4) / (3 * (np.mean(obs_dist ** 2) ** 2)) - 1
                second_moments.append(second_moment)
                alpha2s.append(alpha2)
                group_labels.append(group)

        ax.set_title(f'{group}')
        #ax.set_ylim(0.001, 10)
        #ax.set_xlim(-0.25, 3)
        ax.set_xlabel('Step Size (um)')
        ax.set_yscale('log')  # Set y-axis to logarithmic scale

        # Add shaded gray area
        ax.fill_between(x=[-0.25, 0.0928], y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], color='gray', alpha=0.3)
        ax.legend(fontsize='21', loc='upper right', bbox_to_anchor=(1.5, 1))  # Adjust font size and location
        plt.savefig(f"{plot_dir}/{group}_ss.png", bbox_inches='tight')
        plt.close(fig)

    # Plot the estimator of the second moment for each distribution with group colors
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    scatter_width = 0.2  # Adjust the width of the scatter points

    for i, group in enumerate(my_groups):
        color = cmap(i / (ngroups - 1), 1)  # Use full alpha for legend
        indices = [index for index, label in enumerate(group_labels) if label == group]
        medians = [np.median(obs_dist) for obs_dist in [np.asarray(df[(df[group_col] == group) & (df['tlag'] == tlag)].loc[:, "0":str(stop_pos)]).flatten() for tlag in my_tlags]]

        # Calculate the x-coordinates for the scatter points based on group and width
        x_coordinates = [i + index * scatter_width for index in range(len(indices))]

        plt.scatter(x_coordinates, [alpha2s[index] for index in indices], label=f'Group {group}', color=color, s=50)  # Adjust the size 's' as needed

    plt.xlabel('Group')
    plt.ylabel('Alpha2 Parameter for Each Distribution')

    # Set the x-axis ticks to the middle of each group of scatter points
    tick_positions = [i + ((len(my_tlags) - 1) * scatter_width) / 2 for i in range(ngroups)]
    plt.xticks(tick_positions, my_groups, rotation=45)  # Rotate the x-axis labels by 45 degrees
    plt.savefig(f"{plot_dir}/alpha2.png", bbox_inches='tight')
    plt.close(fig)
    print("Step size distribution analysis completed")

#Alpha vs. LogD (Function 3)
def alpha_logD(home_dir):
    print("Alpha vs. log D analysis started")
    data_dir = f'{home_dir}/Results_250ms'
    file_pattern = "all_data.txt"
    #make a directory to save the plots
    plot_dir = os.path.join(home_dir, "Further_Analyses")
    os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist
    file = glob.glob(f"{data_dir}/{file_pattern}")
    print(file)

    # Initialize an empty list to store individual DataFrames
    dfs = []
   
    # Initialize an empty set to store unique groups
    all_groups = set()

    # Read the input values from the CSV file
    input_values_df = pd.read_csv(f"{data_dir}/input_files.csv")
   
    # Assuming the CSV file has columns "File name" and "ROI"
    input_values_df.columns = ["file name", "roi"]

    raw_df = pd.read_csv(file[0], sep='\t', index_col=0)
   
    # Filter the DataFrame based on matching values in 'file name' and 'roi'
    filtered_df = raw_df.merge(input_values_df, on=["file name", "roi"])
    #df now contains only the rows where both "file name" and "roi" match the input values

    # Update the set with unique groups in each file
    all_groups.update(filtered_df['group'].unique())

    # Append the DataFrame to the list
    dfs.append(filtered_df)

    # Print all unique groups
    print("All unique groups:", all_groups)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    all_data = combined_df  ###not really necessary
    # Drop rows with missing values in 'Column1'
    all_data = all_data.dropna(subset=['roi'])
    
    all_data.reset_index(drop=True)

    # Convert the 'roi' column to string (str) data type
    all_data['roi'] = all_data['roi'].astype(str)
    all_data['fileid'] = all_data.apply(lambda x: x['file name'].split('.')[0].split('_')[-1], axis=1)

    # Create the 'unique_roi' column
    all_data['unique_roi'] = all_data['group'] +'-' + all_data['fileid'] + '-' + all_data['roi']

    # Your existing code
    all_data['log_D'] = np.log(all_data['D'])

    # Get unique groups
    unique_groups = all_data['group'].unique()

    # Define custom sequential color palette based on Viridis
    custom_palette = sns.color_palette("viridis", n_colors=len(unique_groups))

    # Loop over unique groups
    for i, group in enumerate(unique_groups):
        group_data = all_data[all_data['group'] == group]

        # Create a figure and axis for each group
        fig, ax = plt.subplots(figsize=(6, 6))

        """ # Scatter plot with varying shades of the same color for small dots, DON'T HAVE TO RUN BOTH SCATTER AND KDE (heatmap): CHOOSE ONE
        unique_roi_count = len(group_data['unique_roi'].unique())
        for j, unique_roi in enumerate(group_data['unique_roi'].unique()):
            roi_data = group_data[group_data['unique_roi'] == unique_roi]
            color = custom_palette[i]

            # Set transparency based on the position in the list of unique ROIs
            alpha_factor = (j + 1) / unique_roi_count
            sns.scatterplot(data=roi_data, x='log_D', y='aexp', s=10, ax=ax, color=color, alpha=alpha_factor, legend=False) """

        # KDE plot with varying shades of the same color for small dots
        cmap = sns.light_palette(custom_palette[i], as_cmap=True)
        sns.kdeplot(data=group_data, x='log_D', y='aexp', cmap=cmap, fill=True, thresh=0, ax=ax, alpha=0.5, legend=False)

        # Median per cell for centroids
        centroids = group_data.groupby('unique_roi').agg({'log_D': 'median', 'aexp': 'median'})

        # Create a custom colormap for varying shades of the same color for centroids
        colors_centroids = sns.light_palette(custom_palette[i], n_colors=len(centroids))

        # Scatter plot for centroids with varying shades of the same color
        ax.scatter(centroids['log_D'], centroids['aexp'], s=20, c=colors_centroids, edgecolor='black', marker='o')
        
        # Set axis labels
        ax.set_xlabel('$log(D_{100ms})$', fontsize=24)
        ax.set_ylabel('α', fontsize=36)

        # Set horizontal line at y=1
        ax.axhline(y=1, linewidth=1, linestyle='--', color='grey')
        ax.text(-3.7, 1.05, f"$α=1$", rotation=0, color='grey',fontsize=24)

        # Set title
        ax.set_title(f'Group {group}', fontsize=24)

        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=18)

        # Set limits for x and y axes (can add limits if need to standardize)
        #ax.set_xlim(-4, 1)
        #ax.set_ylim(0, 1.5)

        plt.savefig(f"{plot_dir}/{group}_alpha-logD.png")

        #save distributions to csv
        all_data.to_csv(f"{plot_dir}/all_data_filtered.csv", index=False)
    
    print("Alpha vs. log D analysis completed")

if __name__ == "__main__":
    #argparse
    ap = argparse.ArgumentParser()
    requiredGrp = ap.add_argument_group('required arguments')
    requiredGrp.add_argument("-i",'--home_dir', required=True, help="parent folder location")
    args = vars(ap.parse_args())
    home_dir = args['home_dir']
    # Create processes for each function
    p1 = multiprocessing.Process(target=angle_distributions(home_dir))
    p2 = multiprocessing.Process(target=ss_distributions(home_dir))
    p3 = multiprocessing.Process(target=alpha_logD(home_dir))

    # Start the processes
    p1.start()
    p2.start()
    p3.start()

    # Wait for all processes to finish
    p1.join()
    p2.join()
    p3.join()

    print("All functions have completed.")
