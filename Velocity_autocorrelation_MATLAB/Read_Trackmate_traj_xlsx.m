% This script is for extracting trajectory information from TrackMate,
% saved as 'xlsx' files.

function [Trajectory, Frame, x, y, z] = Read_Trackmate_traj_xlsx(filename)
opts = detectImportOptions(filename);

Name = opts.SelectedVariableNames;
Traj_index = find(strcmp(Name,'x_particle__id'));
Frame_index = find(strcmp(Name,'x_particle_detection__t'));
x_index = find(strcmp(Name,'x_particle_detection__x'));
y_index = find(strcmp(Name,'x_particle_detection__y'));
z_index = find(strcmp(Name,'x_particle_detection__z'));
total_index = [Traj_index,Frame_index,x_index,y_index,z_index];

opts.SelectedVariableNames = total_index; % Extract 'particle id; particle detection t; x; y; z' information
M_select = readmatrix(filename,opts);

Trajectory = M_select(:,1); %Trajectory index for certain particle
Frame = M_select(:,2); %Frame index that tractory is detected
x = M_select(:,3); %Particle trajectory x in micron unit, differen from pixel values in Mosaic plugin
y = M_select(:,4); %Particle trajectory y in micron unit
z = M_select(:,5); %Particle trajectory z in micron unit

end

