function [MSD, std_msd, n_MSD]  = calculate_MSD_modi_Tong(x,y,z,dt,conv)

% tau_max = length(x);
tau_max = length(x)-1; %modified by Tong on 9/5/20, due to MSD involving displacement difference

if z == 0
    z = zeros(size(x));
end

MSD = zeros(1,tau_max);

% 
% for tau = 1:tau_max; 
%    sqdisp_x = conv^2*((x(1+tau:end) - x(1:end-tau)).^2); 
%    sqdisp_y = conv^2*((y(1+tau:end) - y(1:end-tau)).^2);
%    sqdisp_z = conv^2*((z(1+tau:end) - z(1:end-tau)).^2);
%    sqdisp = sqdisp_x + sqdisp_y + sqdisp_z;
%    n_MSD = length(sqdisp_x) + length(sqdisp_y) + length(sqdisp_z);
%    MSD(tau) = mean(sqdisp);
% end

% Using weighted means now

std_msd = zeros(1,tau_max);
n_MSD = zeros(1,tau_max);

for tau = 1:tau_max
   sqdisp_x = conv^2*((x(1+tau:end) - x(1:end-tau)).^2); 
   sqdisp_y = conv^2*((y(1+tau:end) - y(1:end-tau)).^2);
   sqdisp_z = conv^2*((z(1+tau:end) - z(1:end-tau)).^2);
   sqdisp = sqdisp_x + sqdisp_y + sqdisp_z;
   n_MSD(tau) = length(sqdisp_x);
   MSD(tau) = mean(sqdisp);
   delta = sqdisp - MSD(tau);
   M2 =  sum(delta.^2);
   std_msd(tau) = sqrt(M2 / n_MSD(tau));
   
end


end