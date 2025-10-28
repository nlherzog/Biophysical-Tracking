% function [MSD_TE, spots_MeanI_low, spots_MeanI_high] = Ensemble_time_MSD_with_intensity_requirements(dt,conv)
dt = 0.05; conv = 1;
% This function is modified from "Ensemble_vs_time_D.m" located at 
% "Lab project 2020/Mitotisc spindle Gem diffusion program/Tong added/Particle_tracking_vs_Intensity",
% which added an additional feature with selecting particle trajectories
% of certain requirements that can be specificed based on the information
% from "spot_..._spots_modi.mat" file acquired from droplet spots analyses.

% --------------------!!!! Important !!!!----------------------------------
% Before running this function, Run "spots_information_combined.m"
% script to know the particle intensity requirements to include certain
% trajectories, i.e., identify parameter values: spots_MeanI_low &
% spots_MeanI_high as well as convert the modified information of tracks
% into modified spots information.
% ------------------------------------------------------------------------

% The underlying assumption is that the trajectory length values within
% "result" from tracked.m should be 1:1 correspondance as
% "spots_modi.L_tracks". This is essential to check since 0 values in x/y
% position in tracks.xlsx requires modification in tracks and shift the 
% labels in trackID and might not correspond to spots.L_tracks. There are
% occassional differences mostly due to tracked.m data but they are
% referring to the same trajectory

% dt=0.01;
% conv is originally the pixel size in the unit of um, used for Mosaic
% tracking but will be 1 since Trackmate tracking has already included um
% in their length unit. For Trackmate tracking, conv=1!
% (used for Mosaic tracking, not useful for Trackmate trakcing
% conv=0.0928571 (100x TIRF without Spindle);or conv=0.065 (100x TIRF with Spindle);
% or conv=0.1342 (60x CONFOCAL))

% Example of using as "[Total_D_inst, Total_D_ens_inst] =
% Ensemble_vs_time_D(0.01,~)"

% Navigate to a directory with tracked_*.mat results files, which is the
% direct extract without linear fitting from trajectory csv files

disp('Select tracked*.mat files for calculating Ensemble vs time average D')
[filename,path] = uigetfile('multiselect','on','tracked*.mat','Select the tracked files to convert');
 cd(path)

Total_traj_length = [];
f_power = fittype('b*x^a+c','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b','c'});

L_cutoff = 10; % Cutoff length for trajectory, only consider trajectories (or MSD+1) with length>L_cutoff (or length>=(MSD_cutoff+1)+1 ), default value is 30;
Fit_cutoff = 10; % Fitting cutoff length for selected trajectories
spots_MeanI_low = MeanInt_BG; % Lower bound for spots intensity requirements
spots_MeanI_high = 900+MeanInt_BG; % Higher bound for spots intensity requirements

% Find out how many files are within the selection as N_files, include the
% case where only one file is selected.
tic
if iscell(filename)
    N_files = length(filename);
else
    N_files = 1;
end

parfor i = 1:N_files % Loop through different files
    
    if N_files == 1
        disp(filename)
        result = importdata(filename);
        filename_main_string = extractBetween(filename,'tracked_','_Tracks');
        result_spots = importdata(['spots_',char(filename_main_string),'_spots_modi.mat']);
    else
        disp(filename{i})
        result = importdata(filename{i});
        filename_main_string = extractBetween(filename{i},'tracked_','_Tracks');
        result_spots = importdata(['spots_',char(filename_main_string),'_spots_modi.mat']);
    end

    for j = 1:length(result) % Loop through different trajectories within the file
        traj_length = length(result(j).tracking.x);
        if (result_spots.Mean_I_modi(j) >= spots_MeanI_low) && (result_spots.Mean_I_modi(j) <= spots_MeanI_high) % Include only particles with certain mean intensity requirements
            Total_traj_length = [Total_traj_length; traj_length];
        end
    end
end

max_track = max(Total_traj_length); % Extract the maximum length of trajectory, which probably won't be 400.
N_ensemble_traj = length(nonzeros(Total_traj_length > L_cutoff)); % Number of trajectories with length > L_cutoff
MSD_ensemble_time_traj = zeros(N_ensemble_traj, max_track-1);

index = 0;
for i = 1:N_files % Loop through different files
    
    if N_files == 1
        disp(filename)
        result = importdata(filename);
        filename_main_string = extractBetween(filename,'tracked_','_Tracks');
        result_spots = importdata(['spots_',char(filename_main_string),'_spots_modi.mat']);
    else
        disp(filename{i})
        result = importdata(filename{i});
        filename_main_string = extractBetween(filename{i},'tracked_','_Tracks');
        result_spots = importdata(['spots_',char(filename_main_string),'_spots_modi.mat']);
    end

    for j = 1:length(result) % Loop through different trajectories within the file
        time_traj = result(j).tracking.time;
        if length(time_traj) <= L_cutoff || (result_spots.Mean_I_modi(j) < spots_MeanI_low) || (result_spots.Mean_I_modi(j) > spots_MeanI_high)
            continue
        else
            index = index+1;
            MSD_ensemble_time_traj(index, 1:length(time_traj)) = result(j).tracking.MSD;
        end
    end
end

% Ensemble-time average of MSD of all trajectories
MSD_TE = zeros(1,max_track-1);
for i = 1:max_track-1
    temp = MSD_ensemble_time_traj(:,i);
    MSD_TE(i) = mean(temp(temp>0));
end

% Power law fitting with time and MSD_TE:
[power_fit,gof] = fit((1:Fit_cutoff)'*dt,MSD_TE(1:Fit_cutoff)',f_power,'display','off','StartPoint',[0,0,0]);
alpha = power_fit.a;
D_500ms = power_fit.b/4;

figure
hold on
plot((1:L_cutoff)*dt,MSD_TE(1:L_cutoff),'o')
plot((1:0.1:L_cutoff)*dt,feval(power_fit,(1:0.1:L_cutoff)*dt),'--')
disp(['alpha = ',num2str(alpha),' & D_{ens-500ms} = ',num2str(D_500ms),'um^2/s'])
% text(0.005,0.1,['\alpha = ',num2str(alpha)],'fontSize',15)
% text(0.005,0.05,['D_{ens-100ms} = ',num2str(D_100ms),'\mum^2/s'],'fontSize',15)
xlabel('Time / s')
ylabel(['$<MSD_{T\geq ',num2str(L_cutoff+1),'\Delta t}>_E$'],'Interpreter','latex')
box on
set(gca,'FontSize',15)
set(gca,'xScale','log')
set(gca,'yScale','log')
% xlim([0.01,1])

% end





