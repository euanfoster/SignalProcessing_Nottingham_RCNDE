%% An accoustic emission experiment 
close all
clear 
%% load data
x1 = load('ex2_2_xt.txt');
x2 = load('ex2_2_yt.txt');
fs = 40e3;
t = (0:1/fs:(1/fs)*length(x1)-(1/fs))';

%% plot signals 
figure(100)
plot(t,x1)
xlabel('Time [s]')
ylabel('Amplitude [V]')
xlim([0 max(t)])
figure(110)
plot(t,x2)
xlabel('Time [s]')
ylabel('Amplitude [V]')
xlim([0 max(t)])

%% FFT 
%perform fft of signal to ensure that white noise is distrubuted across
%sampling frequency 

x1_freq = abs(fft(x1));
x1_freq = x1_freq/max(x1_freq);
x1_freq = mag2db(x1_freq);
f=1e-3*(0:fs/length(x1_freq):fs-fs/length(x1_freq));

figure(200)
plot(f(1:end/2),x1_freq(1:end/2))
xlabel('Frequency [kHz]')
ylabel('Amplitude [dB]')


x2_freq = abs(fft(x2,2056));
x2_freq = x2_freq/max(x2_freq);
x2_freq = mag2db(x2_freq);
f=1e-3*(0:fs/length(x2_freq):fs-fs/length(x2_freq));

figure(210)
plot(f(1:end/2),x2_freq(1:end/2))
xlabel('Frequency [kHz]')
ylabel('Amplitude [dB]')

%% Standard Deviation of x1

sd = std(x1);
fprintf('Standard deviation of x1 :%d\n',sd);


%% Auto Correlate 

%x1_acf = crcorr_norm(x1,x1);
%x1_acf = autocorr(x1);
%matlab autocorr function provides 
figure (300)
x1_acf=crcorr_norm(x1,x1);
lags = (-1023:1:1023);
plot(lags,x1_acf);
xlabel('Lag')
ylabel('Sample Auto Correlation')
xlim([min(lags) max(lags)]);


%% Cross Correlate
x1_x2_cr = crcorr_norm(x1,x2);
figure(400)
plot(lags,x1_x2_cr);
xlabel('Lag')
ylabel('Sample Cross Correlation')
xlim([min(lags) max(lags)]);
