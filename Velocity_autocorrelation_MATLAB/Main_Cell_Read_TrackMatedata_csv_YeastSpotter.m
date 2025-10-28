%% This script is the main function to call "Read_Trackmatedata_csv_YeastSpotter.m" for processing several condition all at once

% The path of "Read_Trackmatedata_csv_YeastSpotter.m" is within the same
% folder as: "/Users/tongshu/Documents/Lab project 2020/Mitotisc spindle Gem diffusion program/Tong added/Particle_tracking_vs_Intensity/Read_TrackMatedata_csv_YeastSpotter.m"
% The format of naming methods of all files are based on the data from
% Meta: /Volumes/research/holtl02lab/holtl02labspace/Holt_Lab_Members/Tong/Levy diffusion data from Meta/Data on 4:2023/Total GFP_RFP movies

% Add script path to the working directory
addpath(genpath('/Users/tongshu/Documents/Lab project 2020/Mitotisc spindle Gem diffusion program/Tong added/Particle_tracking_vs_Intensity'));
addpath(genpath('/Users/tongshu/Documents/Lab project 2020/Mitotisc spindle Gem diffusion program/Tong added/Matlab_ImageJ_ROI (Matlab program for extracting imageJ ROI)'));

% Add experimental data to the working directory and set it as the current
% working folder
addpath(genpath('D:\Tong\Droplet_movie_Meta\Total GFP_RFP movies - data on 042023'));
cd('D:\Tong\Droplet_movie_Meta\Total GFP_RFP movies - data on 042023');
Conditions = {'HIS3','LST7','NAP1','UBX2','VPS41','YLR402W','YML010W'};

% for i = 1:1
for i = 1:length(Conditions)
    condition = Conditions{i};
    Read_TrackMatedata_csv_YeastSpotter(condition);
end
