% 2-point rheology by calculating Drr(R,tau) (Drr_vector_total) and MSD_2p
% References: 
% (1) J. C. Crocker, et al., Two-point microrheology of inhomogeneous soft materials. Phys. Rev. Lett. 85, 888?891 (2000).
% (2) J. C. Crocker, B. D. Hoffman, ?Multiple?Particle Tracking and Two?Point Microrheology in Cells? in Methods in Cell Biology, (Academic Press, 2007), pp. 141?178.

% input is a list of filtered trajectory csv files, 
% output is the structure data containing Drr_vector_total and MSD_2p for
% each trajectory file as "two_point_microrheology"

% Input parameters:
conv = 0.1833; % pixel size to convert pixel values to actual length in the unit of um
dt = 10*10^(-3); % time interval is 10ms per frame, in the unit of second
% dt = 250*10^(-3); % time interval is 250ms per frame
a = 20*10^(-3); % particle radius in um
delta_R =5; % in the uint of pixel number
R_total = (4:2*1:20); % in the uint of pixel number
% R_total = (50:2*5:150);
delta_t_total = 1:10; % in the unit of frame

disp('Select filtered trajectory files for calculating two-point microrheology:')
[filename,path] = uigetfile('multiselect','on','Traj_*_filtered*.csv','Select the filtered trajectory csv files');
cd(path)

if iscell(filename)
    N_filename = length(filename);
elseif ~iscell(filename)
    N_filename = 1;
end

two_point_microrheology = repmat(struct('filename','', 'Drr',[], 'MSD_2p',[]), numel(N_filename), 1);
% two_point_microrheology = repmat(struct('filename','','Drr',[],'R_Drr',[]), numel(N_filename), 1); % Use this line if particle size is unknown and can only calculate Drr*R rather than MSD_2p
for file_index = 1:N_filename
    if N_filename == 1
        filename_indi = filename;
        two_point_microrheology(file_index).filename = filename;
    else
        filename_indi = filename{file_index};
        two_point_microrheology(file_index).filename = filename{file_index};
    end
    Drr_vector_total = conv^2*main(filename_indi, R_total, delta_R, delta_t_total); % in the unit of um^2
    MSD_2p = 1/a*2*conv*R_total'.* ones(length(R_total),length(delta_t_total)).*Drr_vector_total;
%     R_Drr = 2*conv*R_total'.* ones(length(R_total),length(delta_t_total)).*Drr_vector_total;
    two_point_microrheology(file_index).Drr = Drr_vector_total;
    two_point_microrheology(file_index).MSD_2p = MSD_2p;
%     two_point_microrheology(file_index).R_Drr = R_Drr;
end

%% Short script for calculating statistics from structure dataset: two_point_microrheology
% % Fit with weighted linear regression to get alpha values
% test = cat(3,two_point_microrheology.MSD_2p);
% MSD_2p_mean = mean(test,3,'omitNan');
% MSD_2p_std = std(test,0,3,'omitNan');
% N = sum(~isnan(test),3);
% x = log10(dt*delta_t_total);
% y = log10(MSD_2p_mean);
% sy = MSD_2p_std./sqrt(N);             % column vectors
% w  = 1./(log10(sy).^2);                 % weights = 1/variance
% % Fit: y = b0 + b1*x
% mdl = fitlm(x, y, 'linear', 'Weights', w);   % needs Statistics Toolbox
% b0  = mdl.Coefficients.Estimate(1);
% b1  = mdl.Coefficients.Estimate(2);
% % Plot with error bars + fit + 95% CI
% xg = linspace(min(x), max(x), 200).';
% [yhat, yCI] = predict(mdl, xg, 'Prediction','curve');
% errorbar(x, y, sy, 'o', 'CapSize', 0); hold on
% h = plot(xg, yhat, 'LineWidth', 1.8);
% fill([xg;flipud(xg)], [yCI(:,1);flipud(yCI(:,2))], [0 0 0], ...
% 'FaceAlpha', 0.08, 'EdgeColor', 'none');
% xlabel('log_{10}(Time lag \tau / 1 s)'); ylabel('log_{10}(MSD-2p / 1 \mum^2)');

%% Short script for plotting mean/std values of two_point_microrhoelogy with specified colormap
% test = cat(3,two_point_microrheology.MSD_2p);
% MSD_2p_mean = mean(test,3,'omitNan');
% MSD_2p_std = std(test,0,3,'omitNan');
% N = sum(~isnan(test),3);
% MSD_2p_sem = MSD_2p_std./sqrt(N);
% cmap = jet(length(R_total));
% legend_string = cell(1,kength(R_total));
% for i = 1:length(R_total)
%     errorbar(dt*delta_t_total,MSD_2p_mean(i,:),MSD_2p_sem(i,:),'o-','Color',cmap(i,:));
%     legend_string{i} = ['R = ' num2str(R_total(i))];
% end
% xlabel('Time lag \tau (s)')
% ylabel('MSD-2p (\mum^2)')
% set(gca,'FontSize',15)
% legend(legend_string,'Location','Northwest')
% xlim([0.01,0.1])

%% Main function for processing Drr with a range of R_total, delta_R for a filtered trajectory csv file  
function Drr_vector_total = main(fileName, R_total, delta_R, delta_t_total)

    fprintf('load csv file %s\n', fileName);
    s = get_data_from_csv(fileName);
    disp('load complete');
    
    n_frames = length(fieldnames(s));
    Drr_vector_total = zeros(length(R_total),length(delta_t_total));

    for R_index = 1:length(R_total)
        R = R_total(R_index);
        for delta_t_index = 1:length(delta_t_total) % max value till (n_frames-1)
            delta_t = delta_t_total(delta_t_index);
            tpmsd_vector_t = [];

            for t_second = delta_t:n_frames-1
                t_first = t_second - delta_t;
                tpmsd_vector_t = [tpmsd_vector_t, process_two_frames(t_first, t_second, s, R, delta_R)];
            end

            Drr_vector_total(R_index,delta_t_index) = mean(tpmsd_vector_t);
            fprintf('R = %0.2f; delta_t=%d\n', R, delta_t);
        end
    end
end

%% Select two pairs of particles that are within [R-delta_R, R_delta_R] distance range
function res = process_two_frames(t_first, t_second, s, R, delta_R)
    t_first_key = sprintf('frame_%d', t_first);
    t_second_key = sprintf('frame_%d', t_second);

    res = [];
    p_list_first = table2array(s.(t_first_key));
    [p_list_first_len,~] = size(p_list_first);
    p_list_second = table2array(s.(t_second_key));

    for i = 1:p_list_first_len-1
        for j = i+1:p_list_first_len
            p_i_first = p_list_first(i,:); % Extract [traj_#,x,y] position of particle i
            p_j_first = p_list_first(j,:);
            r_ij = norm(p_i_first(2:3) - p_j_first(2:3)); % Calculate the distance between particle i & j
            if (r_ij >= R - delta_R) && (r_ij <= R + delta_R) && ismember(p_i_first(1),p_list_second(:,1)) && ismember(p_j_first(1),p_list_second(:,1))
                index_i = p_list_second(:,1) == p_i_first(1);
                p_i_second = p_list_second(index_i,:);
                index_j = p_list_second(:,1) == p_j_first(1);
                p_j_second = p_list_second(index_j,:);
                res = [res, process_two_particles(p_i_second, p_j_second, p_i_first, p_j_first)];
            end
        end
    end
end

%% Calculate Drr(R,tau) for each pairs of displacements
function res = process_two_particles(p_i_second, p_j_second, p_i_first, p_j_first)
    unit_vector = (p_i_first(2:3) - p_j_first(2:3)) / norm((p_i_first(2:3) - p_j_first(2:3))); % unit vector along the center line of two particles in the first frame
    delta_i = p_i_second(2:3) - p_i_first(2:3);
    delta_j = p_j_second(2:3) - p_j_first(2:3);
    res = dot(delta_i, unit_vector) * dot(delta_j, unit_vector);
end

%% Load filtered csv files into the data structure for subsequent analyses
function res = get_data_from_csv(fileName)
    data = readtable(fileName);
    data = data(:, {'Trajectory', 'Frame', 'x', 'y'});

    data_struct = struct();
    for frameNum = 0:max(unique(data.Frame))
        frame_data = data(data.Frame == frameNum, :);
        frame_data_update = removevars(frame_data,'Frame'); % remove the colume of "Frame"
        frameNumName = sprintf('frame_%d', frameNum);
        data_struct.(frameNumName) = frame_data_update;
    end
    res = data_struct;
end