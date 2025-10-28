% function  [Total_confine1,Total_confine2,Total_delta_t,Total_max_f_t, MSD_total] = MSD_tau_confinement(dt, conv)
dt = 0.05;
conv = 0.065;
% dt is your time step in the unit of second, in GEM movie capture, dt=0.01
% conv is originally the pixel size in the unit of um, used for Mosaic
% tracking but will be 1 since Trackmate tracking has already included um
% in their length unit
% (used for Mosaic tracking, not useful for Trackmate trakcing
% conv=0.0928571 (100x TIRF without Spindle);or conv=0.065 (100x TIRF with Spindle);
% or conv=0.1342 (60x CONFOCAL))

% This function is used to:
%
% (1) Calculate time-averaged MSD of each trajectory and plot using
% <MSD(tau)>/(4*tau) versus tau to see if there is any plateau at the
% beginning and decreases at larger tau values. Trajectories within each
% file are plotted within one plot. Averaged <MSD(tau)>/(4*tau) among
% trajectories within individual file or among all files within one
% condition can also be plotted using sub-function
% "MSD_tau_averaged_plot.m"
%
% (2) Select long trajectory with certatin cutoff and use sliding window to
% calculate time-averaged MSD within each sliding window and plot
% <MSD(tau)> vs. tau and see if it's linear or confined. If it's confined,
% fit using exponential function to get the confinement size. Requires
% further modifications based on methods from hop diffusion.

disp('Select tracked*.mat files for plotting alpha-D graph')
[filename,path] = uigetfile('multiselect','on','tracked*.mat','Select the tracked files to convert');
cd(path)

Total_traj_length = [];
Total_confine1 = [];
Total_confine2 = [];
Total_delta_t = [];
Total_max_f_t = [];

L_cutoff = 10; % Cutoff length for trajectory, only consider trajectories (or MSD+1) with length>=(L_cutoff+1)+1 (or length>L_cutoff), default value is 10;
L_cutoff_slide = 101; % Cutoff length for trajectory suited for sliding window analyses;
L_window = 51; % Sliding window size (MSD_window+1) of selected trajectory;

% Find out how many files are within the selection as N_files, include the
% case where only one file is selected.
tic
if iscell(filename)
    N_files = length(filename);
else
    N_files = 1;
end

MSD_total(N_files) = struct();
parfor i = 1:N_files % Loop through different files
    
    if N_files == 1
        disp(filename)
        result = importdata(filename);
        MSD_total(i).filename = filename;
%         current = figure('Name',filename);
%         hold on
    else
        disp(filename{i})
        result = importdata(filename{i});
        MSD_total(i).filename = filename{i};
%         current = figure('Name',filename{i});
%         hold on
    end
    
    MSD_total(i).MSDdata = cell(length(result),1);
    Total_traj_length_temp = zeros(1,length(result));
    for j = 1:length(result) % Loop through different trajectories within the file
        time_traj = result(j).tracking.time;
        MSD_traj = result(j).tracking.MSD;
        MSD_total(i).MSDdata{j} = [time_traj, MSD_traj, MSD_traj./time_traj/4]; % 1st column is the tau, 2nd column is time-averaged MSD of that trajectory, 3rd column is MSD/(4*tau)

        traj_length = length(result(j).tracking.x);
        Total_traj_length_temp(j) = traj_length;
%         if traj_length <= L_cutoff
%             continue
%         else
% %             figure(current)
% %             plot(time_traj,MSD_traj./time_traj/4)
%             if traj_length > L_cutoff_slide
%                 x = result(j).tracking.x;
%                 y = result(j).tracking.y;
%                 [confine1, confine2, df_dt, max_f_t] = MSD_confinement(x,y,dt,conv,L_window);
%                 Total_confine1 = [Total_confine1, confine1];
%                 Total_confine2 = [Total_confine2, confine2];
%                 Total_delta_t = [Total_delta_t, df_dt];
%                 Total_max_f_t = [Total_max_f_t, max_f_t];
%             end
%         end
    end
    MSD_total(i).Traj_length = Total_traj_length_temp;
    Total_traj_length = [Total_traj_length, Total_traj_length_temp];
%     xlabel('\tau(s)')
%     ylabel('<MSD(\tau)>/(4*\tau)')
%     xlim([0,0.01*20])
end
toc

MSD_tau_averaged_plot(MSD_total, L_cutoff, dt)
% end

%% MSD_confinement is the sub-function to calculate the confinement based on sliding window of each individual selected trajectories
function [confine1, confine2, df_dt, max_f_t] = MSD_confinement(x,y,dt,conv,L_window)
    N = length(x)-L_window+1;
    MSD_max = zeros(1,N);
%     Adjust_R_expo = [];
%     confinement = [];
    
%     f_linear = fittype('a*x+const','dependent',{'y'},'independent',{'x'},'coefficients',{'a','const'});
%     f_expo = fittype('c*(1-exp(-x/tau))+const','dependent',{'y'},'independent',{'x'},'coefficients',{'c','tau','const'});
% %     figure
% %     hold on
    for k = 1:N
        x_temp = x(k:k+L_window-1);
        y_temp = y(k:k+L_window-1);
        [MSD_temp, ~, ~]  = calculate_MSD_modi_Tong(x_temp,y_temp,0,dt,conv);
        MSD_max(k) = max(MSD_temp);
% %         plot((1:L_window-1)*dt,MSD_temp)
%         [linear_fit,gof_linear] = fit((1:L_window-1)'*dt,MSD_temp',f_linear,'display','off','StartPoint',[0,0]);
%         [exp_fit,gof_exp] = fit((1:L_window-1)'*dt,MSD_temp',f_expo,'display','off','StartPoint',[0,0,0]);
%         if gof_linear.adjrsquare < gof_exp.adjrsquare
%             Adjust_R_expo = [Adjust_R_expo; gof_exp.adjrsquare];
%             confinement = [confinement; exp_fit.c+exp_fit.const];
%         end
    end
    TF1 = islocalmax(MSD_max);
    x_plot = 1:N;
    y_plot = sqrt(MSD_max);
%     figure
%     hold on
%     plot(x_plot, y_plot,'.-')
%     plot(x_plot(TF1), y_plot(TF1),'r*')
%     hold off
%     
    F_temp = fft(y_plot-mean(y_plot));
    P2 = abs(F_temp/N);
%     P2 = dt/N*abs(F_temp).^2; % Calculate power spectrum
    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = 1/dt*(0:floor(N/2))/N;
    TF2 = islocalmax(P1);
    f_local_max = f(TF2);
    df_dt = floor(1/mean(diff(f(TF2)))/dt); % Calculate local maximum frequency interval and transform into time domain
    confine1 = y_plot(TF1); % confine1 include all local maximum of original sqrt(MSD_max)
    [confine2, index_f] = max(P1(TF2)); % confine2 is the largest y values of all local maximum of frequency domain (Fourier transformed) sqrt(MSD_max)
    max_f_t = 1/f_local_max(index_f)/dt; % max_f_t calculates the confine2 corresponded frequency and transform into time domain
%     figure
%     hold on
%     plot(f,P1,'.-')
%     plot(f(TF2), P1(TF2),'r*')
%     text(f_local_max(index_f),confine2,['\leftarrow',num2str(max_f_t)])
%     hold off
end