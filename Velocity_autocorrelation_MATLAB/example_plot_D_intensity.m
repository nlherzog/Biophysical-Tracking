% This script is to extract droplet effective diffusion constant from
% result*.m files and corresponds to the droplet intensity information from
% spots*.m files.
% The underlying assumption is that the trajectory length values within
% "pool.lin.track_length{1,1}(index_temp)" should be 1:1 correspondance as
% "spots.L_tracks(index_temp_relay)"

% Another way to extract this information is to gather spots information
% directly using "spots_information_combined.m" script from the same folder
% and corresponds to the droplet Deff information from result.m file using
% trajectory length >=11 filter index. Remember to check consistency in
% trajectory length information from two sources.

clearvars -except MeanInt_BG
[Trajectory, Frame, x, y, z] = Read_Trackmate_traj_xlsx('sample_1_0-7min_Tracks_modi.xlsx'); % Name of modified Excel file
result_name = dir(fullfile(pwd,'/result_sample_1_0-7min_Tracks_modi.mat'));
spots_name = dir(fullfile(pwd,'/spots_sample_1_droplet_diffusion_0-7min_spots_statistics.mat'));
load(result_name.name)
load(spots_name.name)

index_temp=pool.lin.track_length{1,1}>=11; %Find the particle indexes that are selected for calculating diffusion constants, which is calculated based on modified .xlsx file 
Traj_idx = unique(Trajectory); %Better ways to extract trajectory index in the modified Excel file, could replace the above two lines
index_temp_relay = Traj_idx(index_temp); %Find the actual particle index due to deletion of x,y position being zero
x=spots.Mean_I(index_temp_relay)-MeanInt_BG; %Use the above identified index for corresponding droplet mean pixel intensity 
y=pool.lin.D_lin{1,1};

figure(1)
hold on
plot(x,y,'o')
% xlabel('Droplet total intensity (a.u.)','FontSize', 15)
xlabel('Droplet mean pixel intensity (a.u.)','FontSize', 15)
ylabel('D_{eff} (\mum^2/s)','FontSize', 15)
set(gca,'FontSize',15)
box on

figure(2)
hold on
histogram(x,'Normalization','pdf','BinWidth',3)
% xlabel('Droplet total intensity (a.u.)','FontSize', 15)
xlabel('Droplet mean pixel intensity (a.u.)','FontSize', 15)
ylabel('Probability density','FontSize', 15)
box on
set(gca,'FontSize',15)

% legend({'MGM1_300ms','NAP1_300ms','P506_300ms','TCO89_300ms'},'Interpreter','None')
% legend({'MGM1_2000ms','NAP1_2000ms','P506_2000ms','TCO89_2000ms'},'Interpreter','None')
%%Color: [0,0,1]/[0.30,0.75,0.93]/[0.06,1.00,1.00]/[0.39,0.83,0.07]/[0.93,0.69,0.13]/[1.00,0.41,0.16]