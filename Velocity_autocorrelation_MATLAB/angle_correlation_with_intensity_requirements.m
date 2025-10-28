% function [Mean_cos_angle, sem_cos_angle, cos_angle_t_total, t_cross] = angle_correlation_with_intensity_requirements(dt)
dt = 0.05;
% This function is modified from "angle_correlation.m" located at "Lab
% project 2020/Nuclear GEM method paper/Particle_tracking_for_alpha_D",
% which added an additional feature with selecting particle trajectories
% of certain requirements that can be specificed based on the information
% from "spot_....mat" file acquired from droplet spots analyses.
 
% --------------------!!!! Important !!!!----------------------------------
% Before running this function, Run "spots_information_combined.m"
% script to know the particle intensity requirements to include certain
% trajectories, i.e., identify parameter values: spots_MeanI_low &
% spots_MeanI_high as well as convert the modified information of tracks
% into modified spots information.
% -------------------------------------------------------------------------

% The underlying assumption is that the trajectory length values within
% "result" from tracked.m should be 1:1 correspondance as
% "spots_modi.L_tracks". This is essential to check since 0 values in x/y
% position in tracks.xlsx requires modification in tracks and shift the 
% labels in trackID and might not correspond to spots.L_tracks. There are
% occassional differences mostly due to tracked.m data but they are
% referring to the same trajectory

% dt = 0.05;
% Example of using as "
% angle_correlation(0.05)"

% Check angle correlation by calculating averaged cos(theat) at different
% time intervals
 
% Based on the reference: "Modes of correlated angular motion in
% live cells across three distinct time scales", <cos(theta)>.

%Navigate to a directory with tracked_*.mat results files, which is the
%direct extract without linear fitting from trajectory csv files

%****Comment this section for running with "angle_correlation_modi_Maincall.m"
disp('Select tracked*.mat files for calculating angle correlation functions') %*Uncomment this line for running with maincall
[filename,path] = uigetfile('multiselect','on','tracked*.mat','Select the tracked files to convert'); %*Uncomment this line for running with maincall

cd(path)
Total_traj_length = [];

L_cutoff_low = 10; % or 30: Cutoff length for trajectory, only consider trajectories with length>L_cutoff_low
L_cutoff_high = 401; % Highest length for trajectory, only consider trajectories with length<L_cutoff_high
angle_plot_cutoff = 50; % Angle correlation figure up to 'angle_plot_cutoff' length
spots_MeanI_low = MeanInt_BG; % Lower bound for spots intensity requirements
% spots_MeanI_high = 900+MeanInt_BG; % Higher bound for spots intensity requirements
spots_MeanI_high = 10+MeanInt_BG; % Higher bound for spots intensity requirements

% Find out how many files are within the selection as N_files.
if iscell(filename)
    N_files = length(filename);
else
    N_files = 1;
end

for i = 1:N_files % Loop through different files

    if N_files == 1
        disp(filename)
        filename_main_string = extractBetween(filename,'tracked_','_Tracks');
        result = importdata(filename);
        result_spots = importdata(['spots_',char(filename_main_string),'_Spots_modi.mat']);
    else
        disp(filename{i})
        filename_main_string = extractBetween(filename{i},'tracked_','_Tracks');
        result = importdata(filename{i});
        result_spots = importdata(['spots_',char(filename_main_string),'_Spots_modi.mat']);
    end
    
    for j = 1:length(result) % Loop through different trajectories within the file
        traj_length = length(result(j).tracking.x);
        if (result_spots.Mean_I_modi(j) >= spots_MeanI_low) && (result_spots.Mean_I_modi(j) <= spots_MeanI_high) % Include only particles with certain mean intensity requirements
            Total_traj_length = [Total_traj_length; traj_length];
        end
    end
end

max_track = max(Total_traj_length); % Extract the maximum length of trajectory, which probably won't be 400.
N_angle_traj = length(nonzeros(Total_traj_length > L_cutoff_low & Total_traj_length < L_cutoff_high)); % Number of trajectories with length > L_cutoff
x_angle = zeros(N_angle_traj, max_track);
y_angle = zeros(N_angle_traj, max_track);

index = 0;
for i = 1:N_files % Loop through different files
    
    if N_files == 1
        disp(filename)
        filename_main_string = extractBetween(filename,'tracked_','_Tracks');
        result = importdata(filename);
        result_spots = importdata(['spots_',char(filename_main_string),'_Spots_modi.mat']);
    else
        disp(filename{i})
        filename_main_string = extractBetween(filename{i},'tracked_','_Tracks');
        result = importdata(filename{i});
        result_spots = importdata(['spots_',char(filename_main_string),'_Spots_modi.mat']);
    end
    
    for j = 1:length(result) % Loop through different trajectories within the file
        traj_length = length(result(j).tracking.x);
        if traj_length <= L_cutoff_low || traj_length >= L_cutoff_high || (result_spots.Mean_I_modi(j) < spots_MeanI_low) || (result_spots.Mean_I_modi(j) > spots_MeanI_high)
            continue
        else
            index = index+1;
            x_angle(index, 1:traj_length) = result(j).tracking.x;
            y_angle(index, 1:traj_length) = result(j).tracking.y;
        end
    end
end

% Angle correlation information at different time points from x,y positions
% stored above
cos_angle_t_total = cell(angle_plot_cutoff,1);
for i = 1:angle_plot_cutoff
    cos_angle_t_temp = [];
    for j = 1:N_angle_traj
        % Extract non-zero position information
        x_temp = x_angle(j,:);
        y_temp = y_angle(j,:);
        x_temp = x_temp(x_temp~=0);
        y_temp = y_temp(y_temp~=0);
        % Extract vector (delta_x, delta_y) information under different
        % time intervals (i)
        delta_x = x_temp(1+i:end)-x_temp(1:end-i);
        delta_y = y_temp(1+i:end)-y_temp(1:end-i);
        
        cos_angle_t_temp_j = zeros(1,length(delta_x)-i);
        for k = 1:length(delta_x)-i
            cos_angle = (delta_x(k) * delta_x(k+i) + delta_y(k) * delta_y(k+i))/sqrt(delta_x(k)^2+delta_y(k)^2)/sqrt(delta_x(k+i)^2+delta_y(k+i)^2);
            cos_angle_t_temp_j(k) = cos_angle;
        end
        
        cos_angle_t_temp = [cos_angle_t_temp,cos_angle_t_temp_j];
    end
    cos_angle_t_total{i} = cos_angle_t_temp;
end

% Calculate average cos(theta(tau)) at different interval tau
Mean_cos_angle = zeros(1, angle_plot_cutoff);
sem_cos_angle = zeros(1, angle_plot_cutoff);
for i = 1:angle_plot_cutoff
    temp = cos_angle_t_total{i};
    if find(isnan(temp))
        temp = temp(~isnan(temp));
    end
    Mean_cos_angle(i) = mean(temp);
    sem_cos_angle(i) = std(temp)/sqrt(length(temp));
end

% Calculate the time point at which <cos(theta)> is 0 based on linear
% extrapolation
index_temp = find(Mean_cos_angle<0);
x2 = index_temp(1);
if x2 >=2 
    x1 = x2-1;
    y2 = Mean_cos_angle(x2);
    y1 = Mean_cos_angle(x1);
    t_cross = (x1*y2-x2*y1)*dt/(y2-y1);
else
    y2 = Mean_cos_angle(2);
    y1 = Mean_cos_angle(1);
    t_cross = (y2-2*y1)*dt/(y2-y1);
end
    
% figure
hold on
errorbar((1:angle_plot_cutoff)*dt,Mean_cos_angle,sem_cos_angle,'.-')
if t_cross>=0
    text(t_cross,0,['\leftarrow t_{cross} = ',num2str(t_cross),' s'],'FontSize',15)
else
    disp('There is no angle cross timepoint.')
end
xlabel('Time / s')
ylabel('$<cos\ \theta(t)>_{TE}$','Interpreter','latex')
box on
set(gca,'FontSize',15)

% end