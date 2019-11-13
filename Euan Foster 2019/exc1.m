%% exc1_1
close all
clear all

filename1 = './ex1_1_trend1.txt';
filename2 = './ex1_1_trend2.txt';
filename3 = './ex1_1_trend3.txt';

data1 = importdata(filename1).';
data2 = importdata(filename2).';
data3 = importdata(filename3).';

x = linspace(0,4e-6,length(data1));

[coef1,~,mu1] = polyfit(x,data1,2);
[coef2,~,mu2] = polyfit(x,data2,2);
[coef3,~,mu3] = polyfit(x,data3,2);

trend1 = polyval(coef1,x,[],mu1);
trend2 = polyval(coef2,x,[],mu2);
trend3 = polyval(coef3,x,[],mu3);

data1_proc = data1 - trend1;
data2_proc = data2 - trend2;
data3_proc = data3 - trend3;

figure(01)
subplot(3,1,1)
plot(x*1e6,data1,'g','LineWidth',2)
hold on

plot(x*1e6,data1_proc,'b','LineWidth',2)
hold on

plot(x*1e6,trend1,'r','LineWidth',2)

legend('Trend Present','Trend Removed','Trend')
title('Trend 1')
ylabel('Amplitude [V]')
xlabel('Time [\mu s]')

subplot(3,1,2)
plot(x*1e6,data2,'g','LineWidth',2)
hold on

plot(x*1e6,data2_proc,'b','LineWidth',2)
hold on

plot(x*1e6,trend2,'r','LineWidth',2)

legend('Trend Present','Trend Removed','Trend')
title('Trend 2')
ylabel('Amplitude [V]')
xlabel('Time [\mu s]')

subplot(3,1,3)
plot(x*1e6,data3,'g','LineWidth',2)
hold on

plot(x*1e6,data3_proc,'b','LineWidth',2)
hold on

plot(x*1e6,trend3,'r','LineWidth',2)

legend('Trend Present','Trend Removed','Trend')
title('Trend 3')
ylabel('Amplitude [V]')
xlabel('Time [\mu s]')

sgtitle('Trend Removal')

%% Exc1_2
close all
clear all

prompt = {'How many signals would you like to coherently average'};
title = 'Signals?';
dims = [1 35];
definput = {'1000'};
m = inputdlg(prompt,title,dims,definput);
m = str2double(m);

prompt = {'How many samples would you like to coherently average'};
title = 'Samples';
dims = [1 35];
definput = {'100'};
n = inputdlg(prompt,title,dims,definput);
n = str2double(n);

signals = zeros(n,5);

for i = 1:m
   signals(:,i) = rand(n,1);
end

st_dev_bef = mean(std(signals));
co_ave = (1/m)*sum(signals,2);
st_dev_aft = std(co_ave);

sprintf('The expected value is %.2f. The actual value is %.2f\n', sqrt(m), st_dev_bef/st_dev_aft)

%% Exc 1_3
close all
clear all
x = 1:1024;
t = length(x)/2;
w = 2*pi/t;

y1 = cos(x*w);
y2 = cos(x*w);

prod = y1.*y2;
int_est = sum(prod);

figure(01)
subplot(3,1,1)
plot(x,y1,'R','LineWidth',2)
title('A')
ylabel('Amplitude [V]')
xlabel('Samples')

subplot(3,1,2)
plot(x,y2,'k','LineWidth',2)
title('B')
ylabel('Amplitude [V]')
xlabel('Samples')

subplot(3,1,3)
plot(x,prod,'B','LineWidth',2)
ylabel('Amplitude [V]')
xlabel('Samples')
title('Product')

sgtitle('Signal Correlation by Intergration')

sprintf('The integral of the product is %d', int_est)

%% Exc 1_4
close all
clear all
x = 1:1024;
t = length(x)/2;
w1 = 2*pi/t;
w2 = 2*pi/(length(x)/20);

y1 = cos(x*w1);
y2 = cos(x*w2);

prod = y1.*y2;
int_est = sum(prod);

figure(01)
subplot(3,1,1)
plot(x,y1,'R','LineWidth',2)
title('A')
ylabel('Amplitude [V]')
xlabel('Samples')

subplot(3,1,2)
plot(x,y2,'k','LineWidth',2)
title('B')
ylabel('Amplitude [V]')
xlabel('Samples')

subplot(3,1,3)
plot(x,prod,'B','LineWidth',2)
ylabel('Amplitude [V]')
xlabel('Samples')
title('Product')

sgtitle('Signal Correlation by Intergration')


sprintf('The integral of the product is %d', int_est)

%% Exc 1_5
close all
clear all
x = 1:1024;
t = length(x)/2;
w = 2*pi/t;

y1 = cos(x*w);
y2 = sin(x*w);

prod = y1.*y2;
int_est = sum(prod);

figure(01)
subplot(3,1,1)
plot(x,y1,'R','LineWidth',2)
title('A')
ylabel('Amplitude [V]')
xlabel('Samples')

subplot(3,1,2)
plot(x,y2,'k','LineWidth',2)
title('B')
ylabel('Amplitude [V]')
xlabel('Samples')

subplot(3,1,3)
plot(x,prod,'B','LineWidth',2)
ylabel('Amplitude [V]')
xlabel('Samples')
title('Product')

sprintf('The integral of the product is %d', int_est)

%% exc 1_6
close all
clear all

%CORRELATION

%Initialising Arrays
x_t = zeros(1,1000);
x_t(1,672:772) = 1;
x_t(1,291:391) = 1;
x = 0:length(x_t) - 1;

y_t = zeros(1,1000);
y_t(1,1:100) = 1;
y = 0:length(y_t) - 1;

x_corr_norm = fn_xcorr(x_t,y_t,1);
x_corr = fn_xcorr(x_t,y_t,2);
x2 = -max(x):max(x);
%check with MATLAB function
[r_check,lags] = xcorr(x_t,y_t);

%Plotting cross correlations results
figure(01)
subplot(5,1,1)
plot(x,x_t,'B','LineWidth',2);
title('x(t)')
ylabel('Amplitude [V]')
xlabel('Time Samples')

subplot(5,1,2)
plot(y,y_t,'r','LineWidth',2);
title('y(t)')
ylabel('Amplitude [V]')
xlabel('Time Samples')

subplot(5,1,3)
plot(x2,x_corr_norm,'K','LineWidth',2);
title('With Normalisation')
ylabel('Amplitude [V]')
xlabel('Lag')

subplot(5,1,4)
plot(x2,x_corr,'c','LineWidth',2);
title('Without Normalisation')
ylabel('Amplitude [V]')
xlabel('Lag')

subplot(5,1,5)
plot(lags,r_check,'y','LineWidth',2)
title('Matlab Xcorr Function')
ylabel('Amplitude [V]')
xlabel('Lag')

sgtitle('Cross Correlation')

%AUTO-CORRELATION
y_t = x_t;
auto_corr_norm = fn_xcorr(x_t,y_t,1);
auto_corr = fn_xcorr(x_t,y_t,2);
%Check with MATLAB function
[r_check,lags] = xcorr(x_t,x_t);

figure(02)
subplot(5,1,1)
plot(x,x_t,'B','LineWidth',2);
title('x(t)')
ylabel('Amplitude [V]')
xlabel('Time Samples')

subplot(5,1,2)
plot(y,y_t,'r','LineWidth',2);
title('y(t)')
ylabel('Amplitude [V]')
xlabel('Time Samples')

subplot(5,1,3)
plot(x2,auto_corr_norm,'K','LineWidth',2);
title('With Normalisation')
ylabel('Amplitude [V]')
xlabel('Lag')

subplot(5,1,4)
plot(x2,auto_corr,'c','LineWidth',2);
title('Without Normalisation')
ylabel('Amplitude [V]')
xlabel('Lag')

subplot(5,1,5)
plot(lags,r_check,'y','LineWidth',2)
title('Matlab Xcorr Function')
ylabel('Amplitude [V]')
xlabel('Lag')
sgtitle('Auto Correlation');

%% Exc 1_7
% vary A, B and K for various results to be discussed
close all
clear all

A = 4;                  %Amplitude of Signal 1
B = 4;                  %Amplitude of Signal 2
k = 9;                  %Harmonic Wave Number

Fs = 20e6;              % Sampling frequency - Assumed                  
T = 1/Fs;               % Sampling period       
N = 16;                 % Length of signal
t = (0:N-1)*T;          % Time vector

%establishing signals
n = 0:N-1;

signal_1 = A*cos((k*2*pi*n)/N);
signal_2 = B*sin((k*2*pi*n)/N);

%Fft of signals
A1 = fft(signal_1);
B1 = fft(signal_2);

%(1:L/2+1);

f = 0:Fs/N:Fs-Fs/N;

figure(01)
subplot(2,2,1)
plot(t*1e6,signal_1,'g','LineWidth',2)
title('Sin')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')

subplot(2,2,2)
plot(t*1e6,signal_2,'b','LineWidth',2)
title('Cos')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')

subplot(2,2,3)
plot(f/1e6,real(A1),'k','LineWidth',2)
hold on
plot(f/1e6,imag(A1),':r','LineWidth',2)
legend('Real','Imag')
ylim([-35 35])
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')

subplot(2,2,4)
plot(f/1e6,real(B1),'k','LineWidth',2)
hold on
plot(f/1e6,imag(B1),':r','LineWidth',2)
legend('Real','Imag')
ylim([-35 35])
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')

%% Exc 1_8
close all
clear all

filename1 = './ex1_8_interp.txt';

signal = importdata(filename1).';
a = fft(signal);
b = a(1,1:length(signal)/2);
c = a(1,end/2:end);

%calculating frequency bins and sample time
f_s = 40e6;
n = length(signal);
f = f_s * (0:n-1)/n;
t_s = 1/f_s;
t = (0:n-1) * t_s;

padded_fft = zeros(1,length(a)*4);
padded_fft(1,1:32) = b;
padded_fft(1,end-32:end) = c;

n = length(padded_fft);

f_s_int = 160e6;
f_int = f_s_int * (0:n-1)/n;
t_s_int = 1/f_s_int;
t_int = (0:n-1) * t_s_int;

scale = f_s_int / f_s;

interp_signal = scale*real(ifft(padded_fft));

figure(01)
subplot(4,1,1)
plot(t*1e6,signal,'k','LineWidth',1.5)
title('Under Sampled Signal')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')

subplot(4,1,2)
plot(f/1e6,real(a),'r','LineWidth',1.5)
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')

subplot(4,1,3)
plot(f_int/1e6,real(padded_fft),'b','LineWidth',1.5)
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')

subplot(4,1,4)
plot(t_int*1e6,real(interp_signal),'c','LineWidth',1.5)
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')

sgtitle('FFT Interpolation')

figure(02)
plot(t*1e6,signal, 'b')
hold on
plot(t_int*1e6, interp_signal,':r','LineWidth',2)
xlim([0 max(t*1e6)])
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
title('Raw Signal V Interpolated Signal')
legend('Raw Signal','Interpolated Signal')

%% exc 1_9
close all
clear all

filename1 = './ex1_9.txt';
import = importdata(filename1).';
voltage = import(2,:);
time  = import(1,:);

a = fft(voltage);
mod = abs(a);
mod_short = mod(1,1:end/2);

N = length(time);
Fs = 1/(time(2)-time(1));
f = 0:Fs/N:Fs-Fs/N;

[pks,locs,widths,prov] = findpeaks(mod_short);
B = sort(prov,'descend');
targ1 = B(1,1);
targ2 = B(1,2);
ind1 = find(prov == targ1);
ind2 = find(prov == targ2);

freq1 = f(1,locs(1,ind1));
freq2 = f(1,locs(1,ind2));

filtered_voltage = fn_freq_lowpassfilter(voltage', time', freq1);

sprintf('The frequncy of the ultrasonic signal is %d MHz. The single interfering radio frequency is %e MHz. All other components are white random noise', freq1/1e6, freq2/1e6)

figure(01)
subplot(3,1,1)
plot(time*1e6,voltage,'b','LineWidth',1)

title('Voltage Time Trace')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')

subplot(3,1,2)
findpeaks(abs(mod_short),f(1,1:end/2)/1e6,'MinPeakProminence',100,'Annotate','extents')
title('Fourier Spectra')
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')

subplot(3,1,3)
plot(time*1e6,filtered_voltage,'r','LineWidth',1)
title('Filtered Signal')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
%% exc 1_10
close all
clear all

%Loading in data
filename1 = './ex1_10.txt';
import = importdata(filename1);
voltage = import(:,2);
time  = import(:,1);

%taking FFT of voltage array
a = fft(voltage);
%Calculating the modulus of the fft
mod = abs(a);
%truncating modulus due to symmetry in FFT
mod_short = mod(1:(length(mod)/2), 1);

%Calculating sampling frequency, frequency array and cut
N = length(time);
Fs = 1/(time(2)-time(1));
f = 0:Fs/N:Fs-Fs/N;

%Calculating filter paraemters
cut_off_F = 0.2*Fs;
k_0 = N*cut_off_F/Fs;
k = (f./Fs)*N;

%caulcating filter transfer function
h_full = k_0./(k_0 + (i*k));

%applying nyquist as frequency spectra is only valid up to half the
%sampling freq
h_half = h_full(1,2:end/2+1);

%flip and conjugate the second half of the filter
h_half = flip(conj(h_half));
h = h_full;
h(1,length(h)/2+1:end) = h_half;

%seeing frequency response of filter
figure(01)
freqz(h);
[H,w] =freqz(h);

%normalising and converting to dB scale
H = mag2db(abs(H)/max(H));
f_2 = linspace(0,Fs/2,length(H));

%seeing frequency response of filter
figure(02)
plot(f_2/1e6,H);
ylabel('Amplitude [dB]')
xlabel('Frequency [MHz]')

figure(03)
subplot(3,1,1)
plot(f/1e6,abs(h_full),'b','LineWidth',1)
hold on
plot(f/1e6,abs(h),'r','LineWidth',1)
title('Full Transfer Function')
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')

subplot(3,1,2)
stem(f/1e6,real(h),'g','LineWidth',1)
title('Real Part of Transfer Function')
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')

subplot(3,1,3)
stem(f/1e6,imag(h),'r','LineWidth',1)
title('Imaginary Part of Transfer Function')
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')

figure(04)
subplot(2,1,1)
plot(time*1e6,voltage,'b','LineWidth',1)
title('Voltage Time Trace')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')

subplot(2,1,2)
plot(f(1:end/2)/1e6,mod_short,'r','LineWidth',1)
title('Fourier Spectra')
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')

%convolution in frequency domain
filtered_voltage = ifft(h(:).*a);

figure(05)
subplot(2,1,1)
plot(time*1e6,real(filtered_voltage),'b','LineWidth',1)
hold on
plot(time*1e6,voltage,':r','LineWidth',2)
title('Real Voltage Time Trace')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
legend('Filtered','Original')

subplot(2,1,2)
plot(time*1e6,imag(filtered_voltage),'r','LineWidth',1)
title('Imaginary Voltage Time Trace')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')

%taking FFT of voltage array
b = fft(filtered_voltage);
%Calculating the modulus of the fft
mod_b = abs(b);
%truncating modulus due to symmetry in FFT
mod_short_b = mod_b(1:(length(mod_b)/2), 1);

figure(06)
plot(f(1:end/2)/1e6,mod_short_b)
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')


%% exc 1_11

%assume cut off frequency is 0.2Fs
close all
clear all

filename1 = './ex1_11.txt';
import = importdata(filename1).';

voltage = import(2,:);
time  = import(1,:);

N = length(time);
n = 0:N-1;

Fs = 1/(time(2)-time(1));
F0 = 0.2*Fs;

h = exp(-n.*2*pi*F0/Fs);
filtered_voltage = zeros(1,2*N-1);
filtered_voltage_matlab = conv(voltage,h);

%Implementation of discrete convolution i.e. convolution in the time domain
%https://en.wikipedia.org/wiki/Convolution#Discrete_convolution
%wiki page gives more theory than in course notes
for ii = 1:N
    h = exp(-((ii-1):-1:0).*2*pi*(F0/Fs));
    filtered_voltage(ii) = voltage(1:ii)*h';
end

figure(01)
plot(time*1e6,filtered_voltage(1:N),'b',time*1e6,voltage,'--g','Linewidth',3.0)
hold on
plot(time*1e6,filtered_voltage_matlab(1:N),'r')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
legend('Filtered - Self Written Convolution','Original','Filtered - MATLAB Convolution')
title('Time Domain Convolution - Filtering')

%taking FFT of voltage array
a = fft(voltage');
%Calculating the modulus of the fft
mod = abs(a);
%truncating modulus due to symmetry in FFT
mod_short = mod(1:(length(mod)/2), 1);

%taking FFT of voltage array
b = fft(filtered_voltage(1:N)');
%Calculating the modulus of the fft
mod_b = abs(b);
%truncating modulus due to symmetry in FFT
mod_short_b = mod_b(1:(length(mod_b)/2), 1);

f = 0:Fs/N:Fs-Fs/N;

figure(02)
plot(f(1:end/2)/1e6,mod_short)
hold on
plot(f(1:end/2)/1e6,mod_short_b)
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')
legend('Original Spectrum','Filtered Spectrum')
%% exc 1_12
close all
clear all

filename1 = './ex1_12.txt';
import = importdata(filename1).';

voltage = import(2,:);
time  = import(1,:);

N = length(time);
n = 0:N-1;

Fs = 1/(time(2)-time(1));
F0 = 0.2*Fs;

omega_s = 2*pi*Fs;
omega_0 = 2*pi*F0;

CR_T = 2*pi*omega_0/omega_s;

A = 1 + 2*CR_T;
B = 2*CR_T - 1;

x = voltage;
x(end+1) = voltage(end);

y(1) = CR_T; %setting the initial value of Y 
y(2:length(voltage)+1) = 0;

for ii = 2:length(voltage)+1
    y(ii) = 1/A*(x(ii)+x(ii-1)+(B*y(ii-1)));
end

y = y(2:end);

figure(01)
plot(time*1e6,voltage)
hold on
plot(time*1e6,y)
legend('Unfiltered Signal','Filtered Signal')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
title('Z Transformation - Analogue Filter')

%taking FFT of voltage array
a = fft(voltage');
%Calculating the modulus of the fft
mod = abs(a);
%truncating modulus due to symmetry in FFT
mod_short = mod(1:(length(mod)/2), 1);

%taking FFT of voltage array
b = fft(y');
%Calculating the modulus of the fft
mod_b = abs(b);
%truncating modulus due to symmetry in FFT
mod_short_b = mod_b(1:(length(mod_b)/2), 1);

f = 0:Fs/N:Fs-Fs/N;

figure(02)
plot(f(1:end/2)/1e6,mod_short)
hold on
plot(f(1:end/2)/1e6,mod_short_b)
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')
legend('Original Spectrum','Filtered Spectrum')

%% exc 1_13
close all
clear all

filename1 = './ex1_13.txt';
import = importdata(filename1).';

voltage = import(2,:);
time  = import(1,:);
Fs = 1/(time(2)-time(1));

x = voltage;

N = 2;

y(1) = 0;
y(N:length(x)) = 0;

for ii = N+1:length(voltage)
   y(ii) = y(ii-1) + x(ii) - x(ii-N);
end

figure(01)
plot(time*1e6,voltage)
hold on
plot(time*1e6,y)
legend('Unfiltered Signal','Filtered Signal')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
title('Digital Filter - Moving Average')

%taking FFT of voltage array
a = fft(voltage');
%Calculating the modulus of the fft
mod = abs(a);
%truncating modulus due to symmetry in FFT
mod_short = mod(1:(length(mod)/2), 1);

%taking FFT of voltage array
b = fft(y');
%Calculating the modulus of the fft
mod_b = abs(b);
%truncating modulus due to symmetry in FFT
mod_short_b = mod_b(1:(length(mod_b)/2), 1);

N = length(time);
f = 0:Fs/N:Fs-Fs/N;

figure(02)
plot(f(1:end/2)/1e6,mod_short')
hold on
plot(f(1:end/2)/1e6,mod_short_b')
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')
legend('Original Spectrum','Filtered Spectrum')

%% exc 1_14
close all
clear all

filename1 = './ex1_14.txt';
import = importdata(filename1).';

voltage = import(2,:);
time = import(1,:);
Fs = 1/(time(2)-time(1));

y_bandpass = zeros(1,length(voltage));
y_highpass = zeros(1,length(voltage));

x = voltage;

N = 4;  %Arbitraryly picked for filters

%bandpass
for ii = N+1:length(voltage)-N
   y_bandpass(ii) = -y_bandpass(ii-2) + x(ii) - x(ii-4);  
end

%highpass
for ii = N+1:length(voltage)-N
    y_highpass(ii) = -y_highpass(ii-1) + x(ii) - x(ii-4);
end

figure(01)
plot(time*1e6,voltage)
hold on
plot(time*1e6,y_bandpass)
hold on
plot(time*1e6,y_highpass)
legend('Unfiltered Signal','Bandpass','Highpass')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
title('Highpass & Bandpass Filtering')

%taking FFT of voltage array
a = fft(voltage');
%Calculating the modulus of the fft
mod = abs(a);
%truncating modulus due to symmetry in FFT
mod_short = mod(1:(length(mod)/2), 1);

%taking FFT of voltage array
b = fft(y_bandpass');
%Calculating the modulus of the fft
mod_b = abs(b);
%truncating modulus due to symmetry in FFT
mod_short_b = mod_b(1:(length(mod_b)/2), 1);

%taking FFT of voltage array
c = fft(y_highpass');
%Calculating the modulus of the fft
mod_c = abs(c);
%truncating modulus due to symmetry in FFT
mod_short_c = mod_c(1:(length(mod_c)/2), 1);

N = length(time);
f = 0:Fs/N:Fs-Fs/N;

figure(02)
plot(f(1:end/2)/1e6,mod_short')
hold on
plot(f(1:end/2)/1e6,mod_short_b')
hold on
plot(f(1:end/2)/1e6,mod_short_c')
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')
legend('Original Spectrum','Band Pass Spectrum','High Pass Spectrum')

%% exc 1_15
close all
clear all

filename1 = './ex1_15.txt';
import = importdata(filename1).';

voltage = import(2,:);
time  = import(1,:);
Fs = 1/(time(2)-time(1));

%sampling frequency
Fs = 1/(time(2) - time(1));

%P selected for wideband frequency response
%see course notes on notched filter
p = 0.9;

%taget frequency from exc 1.9
theta = 2*pi*(20e6/Fs);

%filter coefficients
a1 = -2*cos(theta);
b1 = 2*p*cos(theta);
b2 = -(p^2);

y_notched = zeros(1,length(voltage));
x = voltage;

%max z coefficient
N = 2;

for ii = N+1:length(voltage)
   y_notched(ii) = b1*y_notched(ii-1) + b2*y_notched(ii-2) + x(ii) + ...
       a1*x(ii-1) + x(ii-2);
end

figure
plot(time*1e6,voltage)
hold on
plot(time*1e6,y_notched)
legend('Unfiltered Signal','Notched Filter')
xlabel('Time [\mu s]')
ylabel('Amplitude [V]')
title('Notch Filtering')

N = length(time);
f = 0:Fs/N:Fs-Fs/N;

%taking FFT of voltage array
a = fft(voltage');
%Calculating the modulus of the fft
mod = abs(a);
%truncating modulus due to symmetry in FFT
mod_short = mod(1:(length(mod)/2), 1);

%taking FFT of voltage array
b = fft(y_notched');
%Calculating the modulus of the fft
mod_b = abs(b);
%truncating modulus due to symmetry in FFT
mod_short_b = mod_b(1:(length(mod_b)/2), 1);

figure(02)
plot(f(1:end/2)/1e6,mod_short')
hold on
plot(f(1:end/2)/1e6,mod_short_b')
xlabel('Frequency [Mhz]')
ylabel('Amplitude [V]')
legend('Original Spectrum','Notched Filtered Spectrum')
title('Notch Filtering - Spectra')