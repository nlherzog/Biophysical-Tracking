% This script is to extract intended trajectory files based on
% "joined.csv", where nuclear ROIs were taken into considerations.

% The input files are from trajectories files in "Raw_traj" folder and 
% "joined.csv". Those raw trajectories files may contain multiple nuclei 
% within a field of view. And "joined.csv" records each trajectory with
% corresponding nuclear ROIs.
% The output files are filtered trajectories, saved in the "Filtered_traj" 
% folder. Each filtered trajectory represent trajectories within one 
% individual nuclear.

disp('Please select Trajectory files for filtering:');
[filename_traj,path_traj] = uigetfile('multiselect','on','.csv','Select Trajectory files for filtering:');
disp('Please select joined.csv file for selecting ROI trajectory:');
[filename_joined_traj,path_joined_traj] = uigetfile('multiselect','off','.csv','Select joined.csv file for filtering:');
addpath(path_traj, path_joined_traj);

cd(path_joined_traj)

[Trajectory, filename, roi] = csvimport(filename_joined_traj , 'columns', {'Trajectory','file name','roi'} ) ;
N_files = length(filename_traj);

for i = 1:N_files 
    disp(filename_traj{i})
    file_idx = find(strcmp(filename, filename_traj{i}));
    traj_idx = Trajectory(file_idx); % Extract trajectory that are within ROIs
    roi_idx = roi(file_idx); % Extract corresponding trajectory ROI
    unique_roi_idx = unique(roi_idx); % Extract unique ROI number
    
    % Read matrix from original csv file, filter using traj_idx and
    % write back into new filtered csv file
    M = readmatrix(filename_traj{i});
    for j = 1:length(unique_roi_idx)
        traj_roi_idx = find(roi_idx == unique_roi_idx(j));
        keep = ismember(M(:,2),traj_idx(traj_roi_idx)); % Second colume is Trajectory # index
        M_filtered = M(keep, :);
        T = array2table(M_filtered); % convert matrix into table to include the head information
        T.Properties.VariableNames = {'index','Trajectory','Frame','x','y','z','m0','m1','m2','m3','m4','NPscore'};
        new_filtered_file_name = convertCharsToStrings(extractBefore(filename_traj{i},".csv")) + "_roi_" + unique_roi_idx(j) + "_filtered.csv";
        writetable(T,new_filtered_file_name)
    end
end