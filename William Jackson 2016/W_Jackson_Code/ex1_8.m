% Use of the FFT interpolation 
close all 
clear 
fs = 40*1e6;%fs known
newfs = 160*1e6; %desired fs known
ratio = newfs/fs; %calculate ratio assume even
raw = importdata('ex1_8_interp.txt'); %import data
raw_fft = fft(raw); %calculate fft
%first section of signal = first half of fft 
int_fft(1:length(raw_fft)/2+1)= raw_fft(1:length(raw_fft)/2+1);
%insert number zeros to create signal of length ratio*original signal in
%the mid point of signal
%length
int_fft((length(raw_fft)/2+1):(length(raw)*ratio)-length(raw_fft)/2) = 0;
%later section of signal equals the second half of the original fft
int_fft(length(int_fft):length(int_fft)+length(raw_fft)/2) = raw_fft(length(raw_fft)/2:end);
%take the ifft
int_ifft = ifft(int_fft);
%take real part of signal (small errors may cause imaginary parts)
% & scale amplitude of signal by the ratio
int_sig = real(ratio*int_ifft);
%create x values for plotting
x =1:length(int_ifft);
x2 = 1:ratio:length(int_ifft);
%plot signals
plot(x,int_sig);
hold on
stem(x2,raw);
xlabel('Index')
ylabel('Amplitude')
legend('Interpolated signal','Raw Signal')
xlim([0 256])