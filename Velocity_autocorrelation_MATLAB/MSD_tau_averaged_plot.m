% This function is used for subsequent analyses from outputs generated in
% "MSD_tau_confinement.m", specifically data "MSD_total" or could be used
% as sub-functions for "MSD_tau_confinement.m"

% It it used to plot the averaged MSD(tau)/(4*tau) from selected
% trajectories with length>L_cutoff within each individual files or
% combined all files within one condition.

function MSD_tau_averaged_plot(MSD_total,L_cutoff,dt) 
    [~,N_files] = size(MSD_total);
    Traj_selected(N_files) = struct();
    N_total_select_traj = 0; % Number of total selected trajectories with length>L_cutoff in all files
    
    for file_index = 1:N_files
        Traj_selected(file_index).filename = MSD_total(file_index).filename;
        Traj_len_temp = MSD_total(file_index).Traj_length;
        index_temp = find(Traj_len_temp>L_cutoff);
        Traj_selected(file_index).select_index = index_temp; % Save the index of trajectories with length>L_cutoff within each file
        N_total_select_traj = N_total_select_traj+length(index_temp);
    end
    
    MSD_tau_combined = zeros(N_total_select_traj,L_cutoff); % Save the 3rd column of MSD_total of all selected trajectories
    flag = 1;
    for file_index = 1:N_files
        index_temp = Traj_selected(file_index).select_index;
        MSD_tau_temp = zeros(length(index_temp),L_cutoff);
        for traj_index = 1:length(index_temp)
            temp = MSD_total(file_index).MSDdata{index_temp(traj_index),1}(:,3);
            MSD_tau_temp(traj_index,:) = temp(1:L_cutoff);
        end
        MSD_tau_combined(flag:flag+length(index_temp)-1,:) = MSD_tau_temp;
        flag = flag+length(index_temp);
        
%         % Here the plot is for averaged selected trajectories within a file
%         figure
%         hold on
%         plot((1:L_cutoff)*dt,mean(MSD_tau_temp),'.-')
%         errorbar((1:L_cutoff)*dt,mean(MSD_tau_temp),std(MSD_tau_temp)/sqrt(length(index_temp)),'*')
%         title(MSD_total(file_index).filename,'Interpreter','None')
%         xlabel('\tau(s)')
%         ylabel('<MSD(\tau)>/(4*\tau)')
%         set(gca,'FontSize',15)
%         hold off
        
    end
    
    % Here the plot is for averaged selected trajectories among all files
    figure
%     shg  % can be used to combined multiple conditions within the same figure
%     hold on
%     errorbar((1:L_cutoff)*dt,mean(MSD_tau_combined),std(MSD_tau_combined)/sqrt(N_total_select_traj),'*-')
    plot((1:L_cutoff)*dt,mean(MSD_tau_combined),'.-')
    errorbar((1:L_cutoff)*dt,mean(MSD_tau_combined),std(MSD_tau_combined)/sqrt(N_total_select_traj),'*')
    title('All selected trajectories')
    xlabel('\tau(s)')
    ylabel('<MSD(\tau)>/(4*\tau)')
    set(gca,'FontSize',15)
    hold off
    
end