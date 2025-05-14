# # when calling the script, use terminal below: 
# python msd_vs_tau.py -i /PATH-TO-HOME-DIRECTORY -f "FILE_PATTERN(Traj_123_abc_GEM*.csv)" -g GROUP

#import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import argparse

def msd_vs_tau(home_dir, file_pattern, group):
    tracks_dir = f'{home_dir}/Tracks/'
    plot_dir = os.path.join(home_dir, "Further_Analyses")
    os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist

    #CHANGE THIS FOR EACH EXPERIMENT
    #file_pattern = "Traj_231025_Uninfected_GEM*.csv"
    #group = "Uninfected"

    #Read the list of specified files from "input_files.csv"
    all_data_df = pd.read_csv(os.path.join(home_dir, "Results", "all_data.txt"), sep="\t")
    input_files_df = pd.read_csv(os.path.join(home_dir, "Results", "input_files.csv"))
    input_files_df.columns = input_files_df.columns.astype(str)


    #set index to combine filename and ROI as one string for unique identifier:
    input_files_df["filename_roi"]=input_files_df["file name"] + "-" + input_files_df["roi"].astype("str")
    input_files_df.set_index("filename_roi", inplace=True) #this changes the index in the dataframe itself to be our unique identifier
    all_data_df["filename_roi"]=all_data_df["file name"] + "-" + all_data_df["roi"].astype("str")
    joined_df = all_data_df[["filename_roi","Trajectory"]].join(input_files_df, on=["filename_roi"], how="right") #subset with just the index (unique identifier) and the Trajectory # to match the trajectory # to ROI
    joined_df.to_csv(os.path.join(home_dir, "Results", "joined.csv"))

    # Get a list of all files matching the pattern
    file_list = glob.glob(tracks_dir + file_pattern)

    # Initialize an empty list to store dataframes
    dfs = []

    # Loop through each file and read it into a DataFrame
    for file_ in file_list:
        df = pd.read_csv(file_)
        filename = os.path.split(file_)[1] #0 = directory, 1 = file name
        joined_file_df = joined_df[joined_df["file name"]==filename]
        #to just get list of trajectories from this particular file in the joined.csv file
        filtered_traj = joined_file_df["Trajectory"]
        filtered_df = df[df['Trajectory'].isin(filtered_traj)]
        dfs.append(filtered_df)

    def fill_track_lengths(track_data):
        # fill track length array with the track lengths
        ids = np.unique(track_data[:, 0])
        track_lengths = np.zeros((len(ids), 2))
        for i,id in enumerate(ids):
            cur_track = track_data[track_data[:, 0] == id]
            track_lengths[i,0] = id
            track_lengths[i,1] = len(cur_track)
        return track_lengths

    #min length of track = 10 for mammalian cells baseline
    def calculate_vacf_average(full_vacf, min_len=10, delta=1):
        if(min_len < 10):
            min_len = 10

        ensemble_average = []

        # filter tracks by length
        valid_tracks = full_vacf[np.where(full_vacf[:, 6] >= (min_len - 1))]

        ids = np.unique(track_lens[track_lens[:, 1] >= min_len][:, 0])

        if(len(valid_tracks)>0):
            max_tlag = int(np.max(valid_tracks[:, 1]))
            #print(max_tlag)
            for tlag in range(0,max_tlag+delta,delta):
                # gather all data for current tlag
                cur_tlag_vacfsX=valid_tracks[valid_tracks[:, 1] == tlag][:,4]
                cur_tlag_vacfsY=valid_tracks[valid_tracks[:, 1] == tlag][:,5]
                ensemble_average.append([tlag,
                                            np.nanmean(cur_tlag_vacfsX),
                                            np.nanstd(cur_tlag_vacfsX),
                                            np.nanstd(cur_tlag_vacfsX)/np.sqrt(cur_tlag_vacfsX.size),
                                            np.nanmean(cur_tlag_vacfsY),
                                            np.nanstd(cur_tlag_vacfsY),
                                            np.nanstd(cur_tlag_vacfsY)/np.sqrt(cur_tlag_vacfsY.size)])
        return np.asarray(ensemble_average)

    def vacf_all_tracks(track_data, track_lens, min_len=10, delta=1):
        # track must be at least length 10
        if (min_len < 10):
            min_len = 10

        filt_track_lens = track_lens[track_lens[:, 1] >= min_len]
        ids = np.unique(filt_track_lens[:, 0])

        VACF=np.zeros(shape=(int(np.sum(filt_track_lens[:, 1])-ids.size), 7))

        i = 0
        for id in ids:
            cur_track = track_data[track_data[:, 0] == id]

            x_v = (cur_track[delta:, 2] - cur_track[:-delta, 2]) / delta
            y_v = (cur_track[delta:, 3] - cur_track[:-delta, 3]) / delta

            cx = np.correlate(x_v, x_v, mode="full")
            cy = np.correlate(y_v, y_v, mode="full")

            VACF[i:i + x_v.size, 0] = id
            VACF[i:i + x_v.size, 1] = delta*np.arange(0, x_v.size, 1)
            VACF[i:i + x_v.size, 2] = cx[cx.size//2:]
            VACF[i:i + x_v.size, 3] = cy[cy.size//2:]

            VACF[i:i + x_v.size, 4] = cx[cx.size // 2:]/cx[cx.size//2]
            VACF[i:i + x_v.size, 5] = cy[cy.size // 2:]/cy[cy.size//2]

            VACF[i:i + x_v.size, 6] = x_v.size
            i+=x_v.size

        # Average correlation at each tau, over all tracks
        # (depending on track length, some tracks will not have data as tau gets bigger)
        # The return value is a matrix with an average autocorrelation value for each tau
        # The number of rows corresponds to the number of tau's available
        # The columns are: avg-VACF(x), std-dev-VACF(x), stderr-VACF(x), avg-VACF(y), std-dev-VACF(y), stderr-VACF(y)
        res = calculate_vacf_average(VACF, min_len, delta)
        return VACF, res

    # Convert to a numpy array
    track_data = df[['Trajectory', 'Frame','x','y']].to_numpy()

    # get track lengths
    track_lens = fill_track_lengths(track_data)

    delta = 1

    # Calculate the velocity autocorrelation (self correlation) for tau=1, for each track
    # Also calculates the average over all tracks
    all_vacf, avg_vacf = vacf_all_tracks(track_data=track_data,
                            track_lens=track_lens,
                            min_len=11,
                            delta=delta)

    colorLight='lightblue'
    colorDark='blue'
    color='purple'

    for delta in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        all_vacf, avg_vacf = vacf_all_tracks(track_data=track_data,
                                            track_lens=track_lens,
                                            min_len=11,
                                            delta=delta)
        
        # plt.scatter(avg_vacf[:, 0], (avg_vacf[:, 1] + avg_vacf[:, 4])/2, alpha=0.5)
        plt.plot(avg_vacf[:, 0], (avg_vacf[:, 1] + avg_vacf[:, 4])/2, alpha=0.5, label=delta, color=colorLight)
        res_df = pd.DataFrame(avg_vacf, columns=['tlag','mean_x','std_x','std_err_x','mean_y','std_y','std_err_y'])
        res_df.to_csv(f"{plot_dir}/vacf_{group}_d{delta}.csv", sep='\t')
    
    plt.axhline(-0.1, color='black', alpha=0.2)
    plt.xlim(0, 100)
    plt.xlabel('Tau', fontsize=36)
    plt.ylabel('VACF', fontsize=36)
    plt.legend()
    plt.savefig(f"{plot_dir}/{group}_vacf.png", bbox_inches='tight')
    plt.close()

    # Track lengths
    counts = df['Trajectory'].value_counts()
    counts = counts.sort_index()
    counts.index

    # Let's define a function for this
    def msd2d(x, y):

        tlags = np.arange(1, len(x), 1)
        MSD = np.zeros(tlags.size)

        for i, shift in enumerate(tlags):
            sum_diffs_sq = np.square(x[shift:] - x[:-shift]) + np.square(y[shift:] - y[:-shift])
            MSD[i] = np.mean(sum_diffs_sq)

        return MSD

    # Make a loop.  compute MSD for each track and save as a pandas data frame
    time_step = 0.01 #change if using different parameters
    voxel = 0.1833

    # Track lengths
    track_lengths = df['Trajectory'].value_counts()
    track_lengths = df.sort_index()

    # Initialize an empty numpy array for storing the data
    # 3 columns: id, tlag, MSD
    track_msd = np.zeros((len(track_data), 3), dtype=float)

    # A list of the track id's
    track_ids = track_lengths.index

    # The number of tracks
    num_tracks = track_lengths.size

    # Loop: the index (i) keeps track of the location we want to add data
    i=0
    for id in track_ids:
        # Select the track using pandas selection syntax
        track = df[df['Trajectory']==id]
        track_length = len(track)
        # Calculate MSD for this track
        msd = msd2d(track['x'].to_numpy() * voxel,
                    track['y'].to_numpy() * voxel)
        # Insert the data into the new numpy array that we created above
        track_msd[i:i+track_length,0] = id
        track_msd[i:i+track_length,1] = np.arange(0,track_length,1)*time_step
        track_msd[i+1:i+track_length,2] = msd # note starts with i+1 here (tlag=0 is msd=0)

        i = i + track_length

    track_msd = pd.DataFrame(track_msd, columns=['Trajectory','tau','MSD'])

    def compute_msd(track_data, voxel=0.1833, time_step=0.01):

        # Track lengths
        track_lengths = track_data['Trajectory'].value_counts()
        track_lengths = track_lengths.sort_index()

        # Initialize an empty numpy array for storing the data
        # 3 columns: id, tlag, MSD
        track_msd = np.zeros((len(track_data), 3), dtype=float)

        # A list of the track id's
        track_ids = track_lengths.index

        # The number of tracks
        num_tracks = track_lengths.size

        # Loop: the index (i) keeps track of the location we want to add data
        i=0
        for id in track_ids:
            # Select the track using pandas selection syntax
            track = track_data[track_data['Trajectory']==id]
            track_length = len(track)

            # Calculate MSD for this track
            msd = msd2d(track['x'].to_numpy() * voxel,
                        track['y'].to_numpy() * voxel)

            # Insert the data into the new numpy array that we created above
            track_msd[i:i+track_length,0] = id
            track_msd[i:i+track_length,1] = np.arange(0,track_length,1)*time_step
            track_msd[i+1:i+track_length,2] = msd # note starts with i+1 here (tlag=0 is msd=0)

            i = i + track_length

        return pd.DataFrame(track_msd, columns=['Trajectory','tau','MSD'])

    all_msds = compute_msd(df)

    # Assuming 'all_msds' is your DataFrame containing individual trajectories

    # Filter trajectories that are more than 10 frames long
    long_trajectories = all_msds.groupby('Trajectory').filter(lambda x: len(x) > 10)
    long_trajectories.to_csv(f"{plot_dir}/{group}_longtraj_msd.csv", index=False)

    # Plot individual trajectories in gray for selected long trajectories
    for traj_id in long_trajectories['Trajectory'].unique():
        traj_data = long_trajectories[long_trajectories['Trajectory'] == traj_id]
        plt.plot(traj_data['tau'], traj_data['MSD'], color='gray', alpha=0.5)

    # Ensemble Averaging for selected long trajectories
    ensemble_avg = long_trajectories.groupby('tau', as_index=False).agg({'MSD': np.mean})

    # Filter out zero or very small values before regression
    filtered_ensemble_avg = ensemble_avg[ensemble_avg['MSD'] > 0]
    filtered_ensemble_avg.to_csv(f"{plot_dir}/{group}_ensemble-avg_msd.csv", index=False)

    # Calculate the slope of the red line (ensemble average)
    slope = 1  # Replace with your desired slope

    # Draw a dashed straight line with the calculated slope for the first half
    x_vals = np.logspace(np.log10(min(filtered_ensemble_avg['tau'])), np.log10(np.median(filtered_ensemble_avg['tau'])), 100)
    x_vals =x_vals[0:50]
    y_vals = 10**(np.log10(filtered_ensemble_avg['tau'].iloc[0]) * slope + np.log10(filtered_ensemble_avg['MSD'].iloc[0]) + 2)
    plt.plot(x_vals, y_vals * (x_vals / x_vals[0]) ** slope, linestyle='dashed', color=colorDark, label=f'Slope: {slope:.2f}')

    # Annotate the plot with the slope value
    plt.annotate(f"Slope: {slope:.2f}", xy=(0.1, 0.7), xycoords="axes fraction", fontsize=10, color=colorDark)

    # Plot ensemble average in red
    plt.plot(ensemble_avg['tau'], ensemble_avg['MSD'], color=colorDark, label='Ensemble Average')

    # Set log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Add labels and legend
    plt.xlabel('Tau')
    plt.ylabel('MSD')
    plt.legend()

    plt.ylim(1e-2, 10)
    plt.savefig(f"{plot_dir}/{group}_msd-tau-all.png", bbox_inches='tight')
    plt.close()

    # Plot only the average
    # Assuming 'all_msds' is your DataFrame containing individual trajectories
    # Filter trajectories that are more than 10 frames long
    long_trajectories = all_msds.groupby('Trajectory').filter(lambda x: len(x) > 10)

    # Ensemble Averaging for selected long trajectories
    ensemble_avg = long_trajectories.groupby('tau', as_index=False).agg({'MSD': np.mean})

    # Filter out zero or very small values before regression
    filtered_ensemble_avg = ensemble_avg[ensemble_avg['MSD'] > 0]

    # Calculate the slope of the red line (ensemble average)
    slope = 1  # Replace with your desired slope

    # Set the size of the plot
    plt.figure(figsize=(6, 6),linewidth=10)  # Adjust the size as needed

    # Draw a dashed straight line with the calculated slope for the first half
    x_vals = np.logspace(np.log10(min(filtered_ensemble_avg['tau'])), np.log10(np.median(filtered_ensemble_avg['tau'])), 100)
    x_vals =x_vals[0:50]
    y_vals = 10**(np.log10(filtered_ensemble_avg['tau'].iloc[0]) * slope + np.log10(filtered_ensemble_avg['MSD'].iloc[0]) + 2)
    plt.plot(x_vals, y_vals * (x_vals / x_vals[0]) ** slope, linestyle='dashed', color=color, label=f'Slope: {slope:.2f}',linewidth=5)

    # Annotate the plot with the slope value
    plt.annotate(f"∝ τ ", xy=(0.05, 0.7), xycoords="axes fraction", fontsize=36, color=color)

    # Plot ensemble average in red
    plt.plot(ensemble_avg['tau'], ensemble_avg['MSD'], color=color, label='Ensemble Average',linewidth=3)

    # Set log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    # Add labels and legend
    plt.xlabel('Tau',fontsize=30)
    plt.ylabel('<δx^2>',fontsize=30)
    plt.legend()

    plt.xlim(1e-2,10)
    plt.ylim(1e-2,10)
    plt.savefig(f"{plot_dir}/{group}_msd-tau-avg.png", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    requiredGrp = ap.add_argument_group('required arguments')
    requiredGrp.add_argument("-i",'--home_dir', required=True, help="parent folder location")
    requiredGrp.add_argument("-f", '--file_pattern', required=True, help="pattern for file matching")
    requiredGrp.add_argument("-g", '--group', required=True, help="group name")
    args = vars(ap.parse_args())
    home_dir = args['home_dir']
    file_pattern = args['file_pattern']
    group = args['group']

msd_vs_tau(home_dir, file_pattern, group)