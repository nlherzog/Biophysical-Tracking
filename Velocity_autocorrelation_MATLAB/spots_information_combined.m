% This small script is to extract combined droplet spots information of
% certain condition based on corresponding *tracks_modi.mat files, which 
% include the modified tracks information due to 0 values in x or y positions

% Remember to save the following variables by adding the condition names:
% "MeanInt_BG, Mean_Intensity_, Mean_Intensity_noBG_, Track_length_,
% filename_"

clearvars -except MeanInt_BG %MeanInt_BG is stored in the Trackmate Background folder
disp('Select spots*.mat files for calculating combined spots information')
[filename,path] = uigetfile('multiselect','on','spots*.mat','Select the spots files to convert');

Track_length=[]; % combinied spots track lengths
Mean_Intensity=[]; % combined spots mean intensity values
Mean_Intensity_noBG=[]; % Extract background intensity from spots mean intensity values

% Find out how many files are within the selection as N_files.
if iscell(filename)
    N_files = length(filename);
else
    N_files = 1;
end

for i = 1:N_files % Loop through different files
    
    if N_files == 1
        disp(filename)
        result_spots = importdata(filename);
        filename_main_string = extractBetween(filename,'spots_','_spots');
    else
        disp(filename{i})
        result_spots = importdata(filename{i});
        filename_main_string = extractBetween(filename{i},'spots_','_spots');
    end
    
    Excel_modi_name = [char(filename_main_string),'_Tracks.xml_modi.xlsx']; % Name of modified Excel file
    [Trajectory, Frame, ~, ~, z] = Read_Trackmate_traj_xlsx(Excel_modi_name); % Extract actual trajectory ID that include skipping of those that have 0 position values
    Traj_idx = unique(Trajectory); %Better ways to extract trajectory index in the modified Excel file
    
    Track_length = [Track_length;result_spots.L_tracks(Traj_idx)];
    Mean_Intensity = [Mean_Intensity;result_spots.Mean_I(Traj_idx)];
    Mean_Intensity_noBG = [Mean_Intensity_noBG;result_spots.Mean_I(Traj_idx)-MeanInt_BG];
    
    spots_modi = struct( 'N_tracks_modi', length(Traj_idx),...
                'L_tracks_modi',result_spots.L_tracks(Traj_idx),...
                'Mean_I_modi',result_spots.Mean_I(Traj_idx),...
                'Max_I_modi',result_spots.Max_I(Traj_idx),...
                'Total_I_modi',result_spots.Total_I(Traj_idx));

    save(['spots_',char(filename_main_string),'_Spots_modi'],'spots_modi')
    
end

figure(1)
hold on
histogram(Mean_Intensity_noBG,'Normalization','pdf')
xlabel('Droplet mean intensity (a.u.)')
ylabel('Probability intensity')
box on
set(gca,'FontSize',15)

figure(2)
hold on
plot(Track_length,Mean_Intensity,'.')
xlabel('Trajectory length')
ylabel('Droplet mean intensity (a.u.)')
box on
set(gca,'FontSize',15)