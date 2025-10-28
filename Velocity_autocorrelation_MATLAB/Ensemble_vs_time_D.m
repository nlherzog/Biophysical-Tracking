function [Total_D_inst, Total_D_ens_inst, MSD_TE] = Ensemble_vs_time_D(dt,conv)
% dt=0.01;
% conv is originally the pixel size in the unit of um, used for Mosaic
% tracking but will be 1 since Trackmate tracking has already included um
% in their length unit
% (used for Mosaic tracking, not useful for Trackmate trakcing
% conv=0.0928571 (100x TIRF without Spindle);or conv=0.065 (100x TIRF with Spindle);
% or conv=0.1342 (60x CONFOCAL))

% Example of using as "[Total_D_inst, Total_D_ens_inst] =
% Ensemble_vs_time_D(0.01,~)"

% (1) Check the ergodicity of particle trajectories by calculating time
% averaging effective diffusion constants (space information) and ensemble 
% averaging effective diffusion constants (time information)
% (2) Calcaulte the ensemble-time averaging of all selected trajectories
 
% Based on the reference: "Non-specific interactions govern cytosolic 
% diffusion of nanosized objects in mammalian cells".

% Navigate to a directory with tracked_*.mat results files, which is the
% direct extract without linear fitting from trajectory csv files

disp('Select tracked*.mat files for calculating Ensemble vs time average D')
[filename,path] = uigetfile('multiselect','on','tracked*.mat','Select the tracked files to convert');
 cd(path)

Total_traj_length = [];
Total_D_inst = [];
Total_D_ens_inst = [];

f_linear = fittype('a*x+b','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b'});
f_power = fittype('b*x^a','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b'});

L_cutoff = 10; % Cutoff length for trajectory, only consider trajectories (or MSD+1) with length>=(MSD_cutoff+1)+1 (or length>L_cutoff), default value is 10;
Fit_cutoff = 10; % Fitting cutoff length for selected trajectories
dn = 2; % Instantaneous diffusion constant is calculated using cali_MSD(dn)/4/dt (0.01s), default value is 2;

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
    else
        disp(filename{i})
        result = importdata(filename{i});
    end

    for j = 1:length(result) % Loop through different trajectories within the file
        time_traj = result(j).tracking.time;
        MSD_traj = result(j).tracking.MSD;
        traj_length = length(result(j).tracking.x);
        Total_traj_length = [Total_traj_length; traj_length];
        if traj_length <= L_cutoff
            continue
        else
            [linear_fit,gof_linear] = fit(time_traj(1:dn),MSD_traj(1:dn),f_linear,'display','off','StartPoint',[0,0]);
            intercept = linear_fit.b;
            D_inst = (MSD_traj(dn)-intercept)/4/time_traj(dn); %<MSD_{dn*dt}>=4*D*(dn*dt)
            Total_D_inst = [Total_D_inst; D_inst];
        end
    end
end

max_track = max(Total_traj_length); % Extract the maximum length of trajectory, which probably won't be 400.
N_ensemble_traj = length(nonzeros(Total_traj_length > L_cutoff)); % Number of trajectories with length > L_cutoff
x_ens = zeros(N_ensemble_traj, max_track);
y_ens = zeros(N_ensemble_traj, max_track);
MSD_ensemble_time_traj = zeros(N_ensemble_traj, max_track-1);

index = 0;
for i = 1:N_files % Loop through different files
    
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
            index = index+1;
            MSD_ensemble_time_traj(index, 1:length(time_traj)) = result(j).tracking.MSD;
            
            fillin_index = result(j).tracking.frame; % frame information starts from 1 (raw frame information starts from 0)
            x_ens(index, fillin_index) = result(j).tracking.x;
            y_ens(index, fillin_index) = result(j).tracking.y;
        end
    end
end

% Ensemble average of MSD information at different time points
for i = 1:max_track-dn
    MSD_ens = zeros(1,dn);
    for j = 1:dn
        idx_temp = find(x_ens(:,i) > 0 & x_ens(:,i+j) > 0);
        MSD_ens(j) = conv^2*mean(((x_ens(idx_temp,i+j) - x_ens(idx_temp,i)).^2 + (y_ens(idx_temp,i+j) - y_ens(idx_temp,i)).^2));
    end
    [linear_fit,gof_linear] = fit((1:dn)'*dt,MSD_ens(1:dn)',f_linear,'display','off','StartPoint',[0,0]);
    intercept = linear_fit.b;
    D_ens_inst = (MSD_ens(dn)-intercept)/4/(dn*dt); %<MSD_{dn*dt}>=4*D*(dn*dt)
    Total_D_ens_inst = [Total_D_ens_inst; D_ens_inst];
end

% Ensemble-time average of MSD of all trajectories
MSD_TE = zeros(1,max_track-1);
for i = 1:max_track-1
    temp = MSD_ensemble_time_traj(:,i);
    MSD_TE(i) = mean(temp(temp>0));
end

% Fitting function choosen from one of below:
% % linear fitting with log(x) and log(y):
% [power_fit,gof] = fit(log((1:Fit_cutoff)'*dt),log(MSD_TE(1:Fit_cutoff)'),f_linear,'display','off','StartPoint',[0,0]);
% alpha = power_fit.a;
% D_100ms = exp(power_fit.b)/4;

% Power law fitting with x and y:
[power_fit,gof] = fit((1:Fit_cutoff)'*dt,MSD_TE(1:Fit_cutoff)',f_power,'display','off','StartPoint',[0,0]);
alpha = power_fit.a;
D_100ms = power_fit.b/4;

% figure
hold on
plot((1:L_cutoff)*dt,MSD_TE(1:L_cutoff),'o')
% plot((1:0.1:L_cutoff)*dt,exp(feval(power_fit,log((1:0.1:L_cutoff)*dt))),'--')
plot((1:0.1:L_cutoff)*dt,feval(power_fit,(1:0.1:L_cutoff)*dt),'--')
disp(['alpha = ',num2str(alpha),' & D_{ens-100ms} = ',num2str(D_100ms),'um^2/s'])
% text(0.005,0.1,['\alpha = ',num2str(alpha)],'fontSize',15)
% text(0.005,0.05,['D_{ens-100ms} = ',num2str(D_100ms),'\mum^2/s'],'fontSize',15)
xlabel('Time / s')
ylabel(['$<MSD_{T\geq ',num2str(L_cutoff+1),'\Delta t}>_E$'],'Interpreter','latex')
box on
set(gca,'FontSize',15)
set(gca,'xScale','log')
set(gca,'yScale','log')
% xlim([0.01,1])

end





