close all 
clear all

%% Loading in Data

clc; %clear command window

disp('Acoustic Emission Analysis'); %display the title

%file name & loading file
filename = 'ex2_2_xt.txt';
voltage1 = importdata(filename);

filename = 'ex2_2_yt.txt';
voltage2 = importdata(filename);

figure
plot(voltage1)
xlabel('Time')
ylabel('Amplitude [V]')
title('Time Trace of X1')

figure
plot(voltage2)
xlabel('Time')
ylabel('Amplitude [V]')
title('Time Trace of X2')

%% Standard Deviation of voltage 1

std_dev = std(voltage1);
fprintf('Standard deviation of x1 :%.5f\n',std_dev);

%% auto correlation of voltage 1
voltage1_corr_norm = fn_xcorr(voltage1',voltage1',1);
x = 0:length(voltage1) - 1;
x2 = -max(x):max(x);

figure(01)
plot(x2,voltage1_corr_norm)
ylabel('Amplitude [V]')
xlabel('Lag')
title('Auto-Correlation of X1')
xlim([min(x2),max(x2)])

%% cross correlation of voltage1 and voltage 2
voltage_xcorr = fn_xcorr(voltage1',voltage2',1);

figure(02)
plot(x2,voltage_xcorr)
ylabel('Amplitude [V]')
xlabel('Lag')
title('Cross-Correlation of X1 & X2')
xlim([min(x2),max(x2)])

location = find(voltage_xcorr == max(voltage_xcorr));
samples = x2(1,location);

fs = 40e3;
delta_t = 1/fs;
time_lag = delta_t*abs(samples);
fprintf('Standard deviation of x1 :%.1f ms \n',time_lag*1e3);

%% Fourier Analysis of Signals
fft1 = fft(voltage1);
fft1 = abs(fft1);
fft1 = fft1(1:end/2,1)';

N = length(voltage1);
f = 0:fs/N:fs-fs/N;
f = f(1,1:end/2);

fft2 = fft(voltage2);
fft2 = abs(fft2);
fft2 = fft2(1:end/2,1)';

figure(03)
plot(f/1e3,fft1)
xlabel('Frequency [KHz]')
ylabel('Amplitude [V]')
title('Fourier Spectra of X1')

figure(04)
plot(f/1e3,fft2)
xlabel('Frequency [KHz]')
ylabel('Amplitude [V]')
title('Fourier Spectra of X2')
