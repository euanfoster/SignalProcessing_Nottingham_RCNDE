% Use of the FFT for noise removal

close all 
clear 
raw = importdata('ex1_9.txt');
%calculate sampling frequency
fs = 1/(raw(2,1)-raw(1,1));
%plot time domain and fft
plot(raw(:,1),raw(:,2))
xlabel('time [s]')
ylabel('Amplitude')
xlim([5e-6 8e-6])
sigpad = raw(:,2);
sigpad(length(raw(:,2)):2068*4)=0;
raw_fft = fft(raw(:,2));
plotfft = fft(sigpad);

f=(0:fs/length(raw_fft):fs-fs/length(raw_fft));
fplot=(0:fs/length(sigpad):fs-fs/length(sigpad));

figure
fft_pos = abs((plotfft(1:end/2+1)));
fft_norm = mag2db(fft_pos/max(fft_pos));
plot(fplot(1:end/2+1),fft_norm);

xlabel('Frequency [Hz]')
ylabel('Amplitude [dB]')
%% Do other things 
%calculate k = f/fs * N
k = f./fs; 
k = k * length(raw_fft);
%k_o = cut off freq f_o/fs * N
f_o = 0.2*fs;
k_o = (f_o/fs)*length(raw_fft);

%implementation of single pole filter
jk = (1j*k);
den = k_o + jk;
H = k_o./den;

H_2 = H(2:length(H)/2+1);
%flip and convolve second half of filter
H_2 = flip(conj(H_2));
H(length(H)/2+1:end) = H_2;
figure
freqz(H);
[h,w] =freqz(H);

h = abs(h);
h = h/max(h);
h = mag2db(h);
axisfs = linspace(0,fs/2,length(h));
figure
plot(axisfs,h);

%%
figure
%convolve filter with signal & plot
fil_fft = H(:).*raw_fft;
stem(abs(fil_fft));
fil_sig = ifft(fil_fft);
rel_sig = real(fil_sig);
figure
plot(raw(:,1),raw(:,2))
hold on
plot(raw(:,1),rel_sig)
xlabel('time [s]')
ylabel('Amplitude')
legend('Raw Signal','Filtered Signal')





