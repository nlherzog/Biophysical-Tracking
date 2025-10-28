% function [VC, VC_sem] = Velocity_correlation(dt)
% dt=0.05;
dt = 0.01;
% This function is to calculate the velocity autocorrelation which can be
% used to distinguish among distinct mechanisms for anomaloous
% sub-diffusion: Cv(tau) = <v(t+tau)*v(t)>; v(t) = (r(t+dt)-r(t))/dt

% Based on the reference: "Vast heterogeneity in cytoplasmic diffusion 
% rates revealed by nanorheology and Doppelgänger simulations"
% & "Analytical Tools To Distinguish the Effects of Localization Error, 
% Confinement, and Medium Elasticity on the Velocity Autocorrelation Function"

%Navigate to a directory with tracked_*.mat results files, which is the
%direct extract without linear fitting from trajectory csv files

disp('Select tracked*.mat files for calculating veloctiy correlation')
[filename,path] = uigetfile('multiselect','on','tracked*.mat','Select the tracked files to convert');
 cd(path)

Total_traj_length = [];

L_cutoff = 10; % Cutoff length for trajectory, only consider trajectories with MSD_length>L_cutoff or traj_length>L_cutoff+1, default values is 10

% Find out how many files are within the selection as N_files.
if iscell(filename)
    N_files = length(filename);
else
    N_files = 1;
end

for i = 1:N_files % Loop through different files to get trajectory length information

    if N_files == 1
        disp(filename)
        result = importdata(filename);
    else
        disp(filename{i})
        result = importdata(filename{i});
    end
    
    for j = 1:length(result) % Loop through different trajectories within the file
        time_traj = result(j).tracking.time; % length of time_traj is the same as length of MSD_traj, which is the same as traj_length-1
        Total_traj_length = [Total_traj_length; length(time_traj)];
    end
end

Max_traj_length = max(Total_traj_length);
N_select_traj = length(nonzeros(Total_traj_length > L_cutoff)); % Number of trajectories with MSD_length > L_cutoff
VC_time_traj = zeros(N_select_traj, Max_traj_length); % Velocity autocorrelation of each trajectory, autocorrelation array also includes tau=0

index = 0;
for i = 1:N_files % Loop through different files to get trajectory length information

    if N_files == 1
        disp(filename)
        result = importdata(filename);
    else
        disp(filename{i})
        result = importdata(filename{i});
    end
    
    for j = 1:length(result) % Loop through different trajectories within the file
        time_traj = result(j).tracking.time;
        if length(time_traj) <= L_cutoff
            continue
        else
            dtau = 2; % This value represents different tau values when calculating velocity autocorrelation, default is 1
            index = index+1;
            VC_time_traj_temp = zeros(dtau, floor(length(time_traj)/dtau)); % Vc_time_traj_temp is a matrix, each row is the autocorrelation of one velocity array. As an example, row 1 is the velocty autocorrelation array of [x(1+dtau)-x(1),x(1+2*dtau)-x(1+dtau),x(1+3*dtau)-x(1+2*dtau),...]
            for array_num = 1:dtau % Loop through "dtau" velocity arrays that have intervals of dtau timelag
                traj_temp_x = result(j).tracking.x(array_num:dtau:end);
                traj_temp_y = result(j).tracking.y(array_num:dtau:end);
                vx_traj = diff(traj_temp_x, 1)/dt/dtau; % calculate x direction velocity using subsequent x displacements
                vy_traj = diff(traj_temp_y, 1)/dt/dtau; % calculate y direction velocity using subsequent y displacements
                if length(vx_traj) > 1 % Only calculate the autocorrelation is there are at least two numbers within the velocity array
                    VC_time_traj_temp(array_num, 1:length(vx_traj)) = (autocorr(vx_traj,'NumLags',length(vx_traj)-1) + autocorr(vy_traj,'NumLags',length(vy_traj)-1))/2;
                end
            end
            VC_time_traj_temp(VC_time_traj_temp == 0) = NaN; % Avoid averaging 0 values typically located at the end of the array
            VC_time_traj(index, 1:floor(length(time_traj)/dtau)) = mean(VC_time_traj_temp,1,'omitnan');
         end
    end
end

% Calculate trajectory-averaged velocity correlation based on individual 
% autocorrelations among all trajectories at different time lag
VC = zeros(1,Max_traj_length);
VC_sem = zeros(1,Max_traj_length);
for i = 1:Max_traj_length
    temp = VC_time_traj(:,i);
    VC(i) = nanmean(temp(temp~=0)); % There exist NAN values in VC_time_traj due to same detection positions over different timepoints
    VC_sem(i) = nanstd(temp(temp~=0))/sqrt(length(temp(temp~=0)));
end

% figure
hold on
% errorbar((0:2*L_cutoff-1)*dt*dtau,VC(1:2*L_cutoff)/VC(1),VC_sem(1:2*L_cutoff)/VC(1),'o')
errorbar((0:length(VC)-1)*dt*dtau,VC/VC(1),VC_sem/VC(1),'o')
xlabel('Time / s')
ylabel('Normalization velocity autocorrelation')
box on
set(gca,'FontSize',15)

% end