% This script is initially adapted from 'Read_TrackMatedata_xml.m' script by
% combining reading droplet spots information with yeast cell mask files
% generated from YeastSpotter to have droplet information (# droplet,
% droplet mean/max/total pixel intensity) within each identified yeast
% cells, named as "Read_TrackMate_xml_YeastSpotter.m'

% We further adapt the code to accept spots information as csv files from
% YeastSpotter particle tracking results and yeast cell identifier as roi 
% files.
% ------------------------------------------------------------------------
% Input: 
% --filename_csv: spots csv files from Trackmate outputs
% --Yeast_roi: ImageJ ROI files including each identified yeast cells,
% needs to be a directory if not in the current folder directly
% --image1: One channel of image for generating spots csv file
% --image2: The other channel of image for extracting droplet pixel intensity inforrmation
%
% Output:
% --Cell_BG1_AVG: cell background based on the cellular ROI median pixel in image1/ channel1
% --Cell_BG2_AVG: cell background based on the cellular ROI median pixel inimage2/ channel2
% 
% Following arrrays are based on the trajectory ID, already added 1 to align with tracking xlsx files
% --Cell_index: cell ROI index for each trajectory, including 0
% --Cell_final_index: trajectory index that have been identified in one and only one particular cellular ROI
% --cell_index_error: trajectory index that have been found in different cellular ROIs
% --MeanInt_cell_noBG1: Mean pixel intensity for every trajectory droplets
% in channel 1 without cellular background, not filterde using Cell_final_index
% --MeanInt_cell_noBG2: Mean pixel intensity for every trajectory droplets in channel 2
% --MaxInt_cell_noBG1: Max pixel intensity for every trajectory droplets in channel 1
% --MaxInt_cell_noBG2: Max pixel intensity for every trajectory droplets in channel 2
% --TotInt_cell_noBG1: Total pixel intensity for every trajectory droplets in channel 1
% --TotInt_cell_noBG2: Total pixel intensity for every trajectory droplets in channel 2

function Read_TrackMatedata_csv_YeastSpotter(condition)
%% Input information is theTrackmate spots csv file name and Yeast_roi directory within working directory

filename_csv = [condition,'_2s_GFP_spots.csv'];
Yeast_roi = ['Images_ROI_from_Meta\Mask_file_from_images\diffusion1_',condition,'_slow_w1cf-Brightfield_t1-ROI.zip']; % Yeast_roi should be a directory if not in the current working folder directory, even if it's within the working path
image1 = [condition,'_slow_GFP.tif'];
image2 = [condition,'_slow_RFP.tif'];
conv = 0.1083330; %Pixel size ('pixelheight') for length and pixel number conversion, in the unit of um

%% Check the presence of folder 'Matlab_ImageJ_ROI' for reading Yeast roi file as structural variables
check = exist('Matlab_ImageJ_ROI*','dir'); % Check the existence of Matlab_imageJ_ROI folder for extracting imageJ ROI files
if check ~= 7
    addpath('/Users/tongshu/Documents/Lab project 2020/Mitotisc spindle Gem diffusion program/Tong added/Matlab_ImageJ_ROI (Matlab program for extracting imageJ ROI)');
%     error('Cannot find Matlab_ImageJ_ROI in the trajectory, need to copy the function into the trajectory')
end

cvsROIs = ReadImageJROI(Yeast_roi); % cvsROIs is a cell array, each cell contains one ROI structure.

%% Extract cellular background based on the images and extracted ROIs, also create a mask file for Matlab in the meantime
image_info1 = imfinfo(image1); % Read the size of the image stacks
image_info2 = imfinfo(image2);
image_stack_size = min(size(image_info1,1),size(image_info2,1)); % In case two images do not have the same z stack size
N_cells = length(cvsROIs);
% Cell background matrix, with each colume as diffrent cell ROIs and each
% row as representative time points
N_represent_time = 5;
Cell_BG1 = zeros(N_represent_time, N_cells);
Cell_BG2 = zeros(N_represent_time, N_cells);
Mask_yeast = zeros(size(imread(image1,1))); % Create a matlab mask image for easier subsequent analyses

for j = 1 : N_cells
    tic
    Coord_temp = cvsROIs{1,j}.mnCoordinates; % line vertices for each determined ROI
    [xq, yq] = meshgrid(1:size(imread(image1,1),1),1:size(imread(image1,1),2)); % create meshgrid vector for the whole image
    in = inpolygon(xq,yq,Coord_temp(:,1),Coord_temp(:,2)); % logical index for any point within the polygon determined by the ROI line vertices
    Mask_yeast(in) = j; % Assign the ROI number to the matlab mask image
    for i = 1 : N_represent_time
        image_temp1 = imread(image1, i*image_stack_size/N_represent_time);
        image_temp2 = imread(image2, i*image_stack_size/N_represent_time);        
        Cell_BG1(i,j) = median(image_temp1(in));
        Cell_BG2(i,j) = median(image_temp2(in));
    end
    toc
end
Cell_BG1_AVG = mean(Cell_BG1); % Cell background of each ROI in GFP channel
Cell_BG2_AVG = mean(Cell_BG2); % Cell background of each ROI in RFP channel

figure
hold on
plot(Cell_BG1_AVG,Cell_BG2_AVG,'.') % Plot the cellular RFP (y) vs GFP (x) background to see if there is any outliner
xlabel('Cellular GFP background (a.u.)')
ylabel('Cellular RFP background (a.u.)')
set(gca,'xScale','log')
set(gca,'yScale','log')
title([condition,'-Background of all cells identified in ROIs'])

%% Extract spots information from structural variable read from .csv file
opts = detectImportOptions(filename_csv);

Name = opts.SelectedVariableNames;
Traj_index = find(strcmp(Name,'TRACK_ID')); 
%'TRACK_ID' in the spots statistics starts from 0 instead of 1, which is 
%1 smaller than 'x_particle__id' in the trajectory xlsx file, which starts
%from 1.
Frame_index = find(strcmp(Name,'FRAME'));
t_index = find(strcmp(Name,'POSITION_T')); %t_index = Frame_index * time_interval
x_index = find(strcmp(Name,'POSITION_X'));
y_index = find(strcmp(Name,'POSITION_Y'));
z_index = find(strcmp(Name,'POSITION_Z'));
%     MeanInt_index = find(strcmp(Name,'MEAN_INTENSITY'));
%     MaxInt_index = find(strcmp(Name,'MAX_INTENSITY'));
%     TotInt_index = find(strcmp(Name,'TOTAL_INTENSITY'));
MeanInt_index = find(strcmp(Name,'MEAN_INTENSITY_CH1'));
MaxInt_index = find(strcmp(Name,'MAX_INTENSITY_CH1'));
TotInt_index = find(strcmp(Name,'TOTAL_INTENSITY_CH1'));
R_index = find(strcmp(Name,'RADIUS'));

total_index = [Traj_index,Frame_index,MeanInt_index,MaxInt_index,TotInt_index,x_index,y_index,z_index,t_index,R_index];
opts.SelectedVariableNames = total_index; %Extract 'Track_ID,Frame,Mean_intensity,Max_intensity,Total_intensity,Position_x,Position_y,Position_Z,Position_T,Radius' information
M_select = readmatrix(filename_csv,opts);

Trajectory = M_select(:,1)+1; %Trajectory index for particles with trajectories, '+1' is to align with trajctory index identified in xlsx trajectory files
Frame = M_select(:,2); %Frame index of identified particles with trajectories, starts from '0'
MeanInt = M_select(:,3); %Mean intensity of identified particles
MaxInt = M_select(:,4); %Maximum intensity of identified particles
TotInt = M_select(:,5); %Total intensity of identified particles
x = M_select(:,6); %Particle trajectory x in micron unit, differen from pixel values in Mosaic plugin
y = M_select(:,7); %Particle trajectory y in micron unit
z = M_select(:,8); %Particle trajectory z in micron unit
t = M_select(:,9); %Particle trajectory t in second unit
r = M_select(:,10); %Particle radius identified in the spots information

%% Combine spots information with yeast cell mask information, colume index is for individual trajectory
trajectory_N = max(Trajectory); % Trajectory index is the key index for comparing with the diffusion information afterwards
Cell_index = zeros(1, trajectory_N); %Cell index for each trajectory
MeanInt_cell_noBG1 = zeros(1, trajectory_N); %Mean intensity of each trajectory averaged through all time points
MeanInt_cell_noBG2 = zeros(1, trajectory_N);
MaxInt_cell_noBG1 = zeros(1, trajectory_N); %Max intensity of each trajectory averaged through all time points
MaxInt_cell_noBG2 = zeros(1, trajectory_N);
TotInt_cell_noBG1 = zeros(1, trajectory_N); %Total intensity of each trajectory averaged through all time points
TotInt_cell_noBG2 = zeros(1, trajectory_N);
cell_index_error = []; % Record any trajectory index that span several cell ROIs

for i = 1:trajectory_N
    disp(['Processing #',num2str(i),' trajectories ...'])
    index_temp = find(Trajectory == i);
    traj_spot_N = length(index_temp); % For each trajectory, extract every spot information within that trajectory
    
    MeanInt_cell_noBG1_temp = zeros(1,traj_spot_N);
    MaxInt_cell_noBG1_temp = zeros(1,traj_spot_N);
    TotInt_cell_noBG1_temp = zeros(1,traj_spot_N);
    MeanInt_cell_noBG2_temp = zeros(1,traj_spot_N);
    MaxInt_cell_noBG2_temp = zeros(1,traj_spot_N);
    TotInt_cell_noBG2_temp = zeros(1,traj_spot_N);
    
    for j = 1:traj_spot_N
        x_temp = x(index_temp(j));  %X_position information
        y_temp = y(index_temp(j));  %Y_position information
        r_temp = r(index_temp(j));  %Radius information

        [MeanInt_temp2,MaxInt_temp2,TotInt_temp2] = Circle_Tool(x_temp,y_temp,r_temp,conv,image2);
        
        cell_index_temp = Mask_yeast(floor(y_temp/conv)+1,floor(x_temp/conv)+1); %Y_position is row, X_position is colume for matrix. '+1' is because matrix index starts with 1. Cell index for the spot j
        %cell_index_temp starts from 1, background pixel value is 0;
        if cell_index_temp > 0 %There are cases where certain droplet trajectory is not within in identified cell ROIs
            MeanInt_cell_noBG1_temp(j) = MeanInt(index_temp(j))-Cell_BG1_AVG(cell_index_temp); %Mean_intensity information
            MaxInt_cell_noBG1_temp(j) = MaxInt(index_temp(j))-Cell_BG1_AVG(cell_index_temp); %Max_intensity information
            TotInt_cell_noBG1_temp(j) = TotInt(index_temp(j))-Cell_BG1_AVG(cell_index_temp)*pi*r_temp^2/conv^2; %Total_intensity information

            MeanInt_cell_noBG2_temp(j) = MeanInt_temp2-Cell_BG2_AVG(cell_index_temp); %Mean_intensity information
            MaxInt_cell_noBG2_temp(j) = MaxInt_temp2-Cell_BG2_AVG(cell_index_temp); %Max_intensity information
            TotInt_cell_noBG2_temp(j) = TotInt_temp2-Cell_BG2_AVG(cell_index_temp)*pi*r_temp^2/conv^2; %Total_intensity information
            
            if j == 1 % Check if each droplet trajectory is within the same cell ROI across all time points
                cell_index_flag = cell_index_temp;
            elseif cell_index_temp ~= cell_index_flag % This is to consider if droplet trajectory is not only located within one cell ROI
                disp(['Trajectory #',num2str(i),' has different cell index at #',num2str(j),' time points as ',num2str(cell_index_temp),' rather than previous ',num2str(cell_index_flag)])
                cell_index_error = [cell_index_error,i];
                break % Exit the current for loop for j index
            end
            
        end
        
    end
    Cell_index(i) = cell_index_temp;
    MeanInt_cell_noBG1(i) = mean(MeanInt_cell_noBG1_temp);
    MeanInt_cell_noBG2(i) = mean(MeanInt_cell_noBG2_temp);
    MaxInt_cell_noBG1(i) = mean(MaxInt_cell_noBG1_temp);
    MaxInt_cell_noBG2(i) = mean(MaxInt_cell_noBG2_temp);
    TotInt_cell_noBG1(i) = mean(TotInt_cell_noBG1_temp);
    TotInt_cell_noBG2(i) = mean(TotInt_cell_noBG2_temp);    
end

% Extract trajectory index that is actually within identified cell ROIs and
% no spanning across different cell ROIs
Cell_final_index = find(Cell_index ~= 0); % get rid of any trajectory located outside cell ROIs
common = intersect(Cell_final_index, cell_index_error);
Cell_final_index = setxor(Cell_final_index, common); % get rid of any trajectory that span different cell ROIs
%% Plot the Droplet intensities information based on each trajectory
% figure('Name',filename_xml)
figure
plot(Cell_BG1_AVG(Cell_index(Cell_final_index)),Cell_BG2_AVG(Cell_index(Cell_final_index)),'.') % Plot the cellular RFP (y) vs GFP (x) background within cells that have droplets identified
xlabel('Cellular GFP background (a.u.)')
ylabel('Cellular RFP background (a.u.)')
set(gca,'xScale','log')
set(gca,'yScale','log')
title([condition,'-Background of cells with droplets'])

% figure('Name',filename_xml)
figure
plot(MeanInt_cell_noBG1(Cell_final_index),MeanInt_cell_noBG2(Cell_final_index),'.') % Plot the mean intensities for droplet RFP (y) vs GFP (x) without cellular background
xlabel('Mean pixel intensity for droplet GFP (a.u.)')
ylabel('Mean pixel intensity for droplet RFP (a.u.)')
set(gca,'xScale','log')
set(gca,'yScale','log')
title([condition,'-Droplet mean pixel intensities without background'])

% figure('Name',filename_xml)
figure
plot(MaxInt_cell_noBG1(Cell_final_index),MaxInt_cell_noBG2(Cell_final_index),'.') % Plot the mean intensities for droplet RFP (y) vs GFP (x) without cellular background
xlabel('Max pixel intensity for droplet GFP (a.u.)')
ylabel('Max pixel intensity for droplet RFP (a.u.)')
set(gca,'xScale','log')
set(gca,'yScale','log')
title([condition,'-Droplet max pixel intensities without background'])

clearvars -except filename_csv Yeast_roi image1 image2 conv Cell_BG1_AVG Cell_BG2_AVG Cell_index Cell_final_index cell_index_error MeanInt_cell_noBG1 MeanInt_cell_noBG2 MaxInt_cell_noBG1 MaxInt_cell_noBG2 TotInt_cell_noBG1 TotInt_cell_noBG2
save(['D:\Tong\Droplet_movie_Meta\Total GFP_RFP movies - data on 042023\Trackmate_Outputs\Slow 2s\',extractBefore(filename_csv,'.csv'),'_CellROI_mask.mat'])

end

%% This is the sub-function to extract droplet pixel information in new images based on the droplets position information
function [Mean_pixel_circle,Max_pixel_circle,Total_pixel_circle] = Circle_Tool(x,y,r,conv,image)
I = imread(image);
% Create circle
th = linspace(0,2*pi);  
% Here +1 is due to pixel index on imageJ starts from 0 (x starts from 0) while matlab matrix index starts from 1
xc = unique(round(x/conv+1+r/conv*cos(th)));
yc = unique(round(y/conv+1+r/conv*cos(th)));
xc = xc(xc<=size(I,2) & xc>0); % Only take points within the frame size
yc = yc(yc<=size(I,1) & yc>0);
I_square = I(yc,xc);

% Mask matrix to get rid of pixels that are outside of circle
I_mask = ones(size(I_square));
Lx = size(I_square,1);
Ly = size(I_square,2);
for i_mask = 1:Lx
    for j_mask = 1:Ly
        distance_temp = sqrt((Lx/2-i_mask)^2+(Ly/2-j_mask)^2);
        if distance_temp > r/conv
            I_mask(i_mask,j_mask) = 0;
        end
    end
end

I_circle = double(I_square).*I_mask;
Mean_pixel_circle = mean(I_circle(I_circle>0));
Max_pixel_circle = max(I_circle(I_circle>0));
Total_pixel_circle = sum(I_circle(I_circle>0),'all');
end