close all 
clear all

%% Importing the data
clc; %clear command window

disp('FREQUENCY-DEPENDENT REFLECTION COEFFICIENT FROM AN COMPOSITE COMPONENT'); %display the title

%file name & loading file
filename = 'ex2_1.txt';
import = importdata(filename);

voltage = import(:,1);
fs = 40e6;
ts = 1/fs;
time = 1:length(voltage);
time = (time*ts)';
thickness = 4e-3;

%% Plotting the initial data
figure(01)
plot(time*1e6,voltage)
ylabel('Amplitude [V]')
xlabel('Time [\mu s]')

%% Windowing the data

%Time gating front and back wall
front_wall_voltage = zeros(length(voltage),1);
front_wall_voltage(40:100,1) = voltage(40:100,1) .* tukeywin(length(voltage(40:100,1)),0.8);
front_wall_time = zeros(length(voltage),1);
front_wall_time(40:100,1) = time(40:100,1);

back_wall_voltage = zeros(length(voltage),1);
back_wall_voltage(146:206,1) = voltage(146:206,1) .* tukeywin(length(voltage(146:206,1)),0.8) ;
back_wall_voltage = back_wall_voltage * (1/0.47); %coursework give reflection coefficient as 0.53 so transmission coefficient is 0.47
back_wall_time = zeros(length(voltage),1);
back_wall_time(146:206,1) = time(146:206,1);

window_plot = zeros(length(voltage),1);
window_plot(40:100,1) = tukeywin(length(voltage(40:100,1)),0.8);
window_plot(146:206,1) = tukeywin(length(voltage(146:206,1)),0.8);

%% Plotting time gating and windowing of signal
figure(02)
plot(time*1e6,voltage,'LineWidth',2)
hold on
plot(front_wall_time*1e6,front_wall_voltage,'LineWidth',2)
hold on
plot(back_wall_time*1e6,back_wall_voltage,'LineWidth',2)
hold on
plot(time*1e6,window_plot,'LineWidth',2)
legend('Original Signal','Windowed Front Wall','Windowed Back Wall','Overall Window Function')
ylabel('Amplitude [V]')
xlabel('Time [\mu s]')
title('Signal Windowing')


%% Calculating FFT of Isolated Waves for Both Cases
%FFT of isolated waves
%Front Wall
n = length(front_wall_voltage);
f_front_wall = fs*(0:(n/2))/n;
Y_front_wall = fft(front_wall_voltage,n);
P_front_wall = abs(Y_front_wall/n);
P_front_wall_B = P_front_wall(1:n/2+1,1);           %Truncating FFT to half length and only storing backwall

%Back Wall
n = length(back_wall_voltage);
f_back_wall = fs*(0:(n/2))/n;
Y_back_wall = fft(back_wall_voltage,n);
P_back_wall = abs(Y_back_wall/n);
P_back_wall_B = P_back_wall(1:n/2+1,1);           %Truncating FFT to half length and only storing backwall
%% Plotting Frequency Spectra of back & front wall
figure(01)
plot(f_front_wall/1e6,P_front_wall_B,'b','LineWidth',2)
hold on
plot(f_back_wall / 1e6, P_back_wall_B,'r','LineWidth',2);
title('Frequency Spectra')
xlabel('Freq (Mhz)');
ylabel('Magnitude');
legend('Front Wall','Back Wall')

%% Calculating Reflection Coefficient

attenuation = (-1/(2*thickness))*log(P_back_wall_B./P_front_wall_B);

%% Plotting Attenuation Coefficient
figure
plot(f_front_wall/1e6,attenuation,'r','LineWidth',2)
title('Attenuation over Frequency')
xlabel('Freq [Mhz]');
ylabel('Attenuation [Nepers/m]');
