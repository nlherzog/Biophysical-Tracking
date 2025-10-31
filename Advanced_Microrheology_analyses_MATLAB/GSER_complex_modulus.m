% This script is to calculate complex viscoelastic modulus based on 
% particle trajectories using generalized Stokes-Einstein relation:
% ~~~~~~~~~~~G^*(\omega)=kbT/[i*\omega*<\Deltar(\omega)^2>*\pi*a]~~~~~~~~~
% <\Deltar(\omega)^2> is the unialteral Fourier transform of MSD
%
% The assumption of using GSER is that tracer is embedded in a completely
% homogeneous materials, and has "no-slip" boundary condition, where fluid
% moleculues at the particle surface "stick" to it and move along with it
% without sliding.
%
% References: 
% (1) T. G. Mason, D. A. Weitz, Optical measurements of frequency-dependent linear viscoelastic moduli of complex fluids. Phys. Rev. Lett. 74, 1250?1253 (1995).
% (2) T. G. Mason, Estimating the viscoelastic moduli of complex fluids using the generalized Stokes-Einstein equation. Rheol. Acta 39, 371?378 (2000).

% Input parameters:
MSD_data_2D = MSD_m2'; % Ensemble-time-averaged mean-square displacement data as a function of time, remember to transform unit from um to m
dt = 0.01; % Time interval for movie in second, 10ms in this case
T = 37+273.15; % Temperature in Kelvin
R = 20e-9; % Diameter in meter, for 40nm (pfV) nucGEM, its diameter is 40nm
kb = 1.380649e-23; % Boltzmann constant in J/K

%% Calculate the Fourier transform of the MSD data based on Mason, 2000 approach
MSD_data_3D = MSD_data_2D/4*6; % Transform 2D MSD data to 3D MSD data
N_MSD = length(MSD_data_3D); % Length of MSD
t = dt*(1:N_MSD)';
alpha = alpha_MSD(MSD_data_3D,t); % Logarithmic derivative of MSD at each point
omega_MSD = 1./((1:N_MSD)'*dt);

G_abs_MSD = kb*T./(pi*R.*(MSD_data_3D(1:end)).*gamma(1+alpha));
% G_abs_MSD = kb*T./(pi.*(MSD_data_3D(1:end)).*gamma(1+alpha)); % if R is unknown, for exmaple histone H2A, calculate G*R rather than G
delta_MSD = pi/2.*diff(log(G_abs_MSD))./diff(log(omega_MSD(1:end)));
G_storage_MSD = G_abs_MSD(1:end-1).*cos(delta_MSD); % Elastic components
G_loss_MSD = G_abs_MSD(1:end-1).*sin(delta_MSD); % Viscous components

% Plot the storage and loss modulus
figure(1)
hold on
plot(omega_MSD(1:end-1), G_storage_MSD,'b--','LineWidth', 1.5);
hold on;
plot(omega_MSD(1:end-1), G_loss_MSD,'r--','LineWidth',1.5);
xlabel('\omega (rad/sec)')
ylabel('Complex shear modulus (Pa)')
% ylabel('Complex shear modulus * R (Pa \cdot m)')
legend({'G''(storage)', 'G"(loss)'},'Location','northwest')
set(gca,'FontSize',15)

% Plot the ratio between storage and loss modulus
figure(2)
hold on
plot(omega_MSD(1:end-1), G_loss_MSD./G_storage_MSD,'--','LineWidth', 1.5);
xlabel('\omega (rad/sec)')
ylabel('Loss modulus / Storage modulus')
set(gca,'FontSize',15)

%% Calculate alpha based on sliding window of MSD vs tau in the logarithm space
function alpha = alpha_MSD(MSD_data_3D,t)
% w = 5; % odd window length in points >=3 for smoothing
w = 1;
if w >= 3
    x = log(t);
    y = log(MSD_data_3D);

    N = length(x);
    h = floor(w/2);
    alpha = zeros(N,1);
    for i = 1+h:N-h
        xi = x(i-h:i+h);
        yi = y(i-h:i+h);
        p1 = polyfit(xi,yi,1);
        alpha(i) = p1(1);
    end

    flag = 0;
    for i = h:-1:2
        flag = flag + 1;
        h_adjust = h-flag;
        xi = x(i-h_adjust:i+h_adjust);
        yi = y(i-h_adjust:i+h_adjust);
        p1 = polyfit(xi,yi,1);
        alpha(i) = p1(1);
    end

    p1 = polyfit(x(1:2),y(1:2),1);
    alpha(1) = p1(1);
    
    flag = 0;
    for i = N-h+1:N-1
        flag = flag + 1;
        h_adjust = h-flag;
        xi = x(i-h_adjust:i+h_adjust);
        yi = y(i-h_adjust:i+h_adjust);
        p1 = polyfit(xi,yi,1);
        alpha(i) = p1(1);
    end
    
    p1 = polyfit(x(end-1:end),y(end-1:end),1);
    alpha(N) = p1(1);

elseif w == 1
        alpha = diff(log(MSD_data_3D))./diff(log(t));
        alpha = [alpha(1);alpha];
end

end