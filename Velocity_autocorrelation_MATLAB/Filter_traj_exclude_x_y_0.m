% This script is used for filtering particle trajectories from Trackmate
% tracking result to exclude any trajectory with position x or y being 0.
% The original Trackmate trajectory file is <filename>.xlsx. After
% filtering the modified Trackmate trajectory file is <filename>_modi.xlsx

% The output <filename>_modi.xlsx is not ready as input for
% result_diffusion_modi_Tong.m due to different table name variables. It
% needs to be opened in Excel and replace the first two rows with the 
% original file first two rows and saved as the final modified input files.

[filename_traj,path_traj] = uigetfile('multiselect','on','.xlsx','Select Trajectory files for filtering:');
addpath(path_traj);
cd(path_traj)

if iscell(filename_traj) 
    N_files = length(filename_traj);
elseif filename_traj
    N_files = 1;
end

for i = 1:N_files 
    if N_files == 1
        disp(filename_traj)
        [Trajectory, Frame, x, y, z] = Read_Trackmate_traj_xlsx(filename_traj) ;
        new_filtered_file_name = extractBefore(filename_traj,".xlsx")+"_modi.xlsx";
        M_table = readtable(filename_traj);
        M_matrix = readmatrix(filename_traj);
    else
        disp(filename_traj{i})
        [Trajectory, Frame, x, y, z] = Read_Trackmate_traj_xlsx(filename_traj{i}) ;
        new_filtered_file_name = extractBefore(filename_traj{i},".xlsx")+"_modi.xlsx";
        M_table = readtable(filename_traj{i});
        M_matrix = readmatrix(filename_traj{i});
    end

    index_0_x_y = union(find(~x),find(~y)); % index of row where either x or y position is 0
    select_traj = setdiff(Trajectory,Trajectory(index_0_x_y)); % Exclude trajectories that have x or y position being 0
    index_traj = ismember(Trajectory,select_traj,'row'); % logical index for all selected rows
    
    % Read table from original xlsx file, filter using index_traj_table and
    % write back into new filtered xlsx file
    M_table(~index_traj,:)=[];
    writetable(M_table,new_filtered_file_name) % At this step, all numbers are converted to text
    % The following steps convert the numbers that were replaced by text
    % back to numbers again
    temp = M_matrix(:,1:2); % colume: frameInterval, frarmeInterval/#agg
    writematrix(temp(index_traj,:),new_filtered_file_name,'Range','A3');
    temp = M_matrix(:,5:6); % colume: nTracks, nTracks/#agg
    writematrix(temp(index_traj,:),new_filtered_file_name,'Range','E3');
    temp = M_matrix(:,9:14); % colume: particle/#id, particle/@nSpots, particle/detection/@t, particle/detection/@x, particle/detection/@y, particle/detection/@z
    writematrix(temp(index_traj,:),new_filtered_file_name,'Range','I3');
end