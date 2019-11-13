% Use of the FFT
%% Generate Signals & FFTS
close all
clear
A = 1;
B = 1;
N = 16;
n = 1:N;
k = 1;
for k = 1:16
    A_cos(k,:) = A*cos(n*k*(2*pi/N));
    B_sin(k,:) = B*sin(n*k*(2*pi/N));
    rel_A(k,:) = fftshift((real(fft(A_cos(k,:)))));
    ima_A(k,:) = fftshift(imag(fft(A_cos(k,:))));
    rel_B(k,:) = fftshift(real(fft(B_sin(k,:))));
    ima_B(k,:) = fftshift(imag(fft(B_sin(k,:))));
    
end
%% Plot
kval = 9;
bins = [-8:1:7];
stem(A_cos(kval,:))

xlabel('n')
ylabel('Amplitude')
figure
stem(bins,rel_A(kval,:))

hold on 
stem(bins,ima_A(kval,:))
xlabel('DFT Bin')
ylabel('Amplitude')
legend('Real Part','Imginary Part')


