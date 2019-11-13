% Convolution in time domain 
close all 
clear 
raw = importdata('ex1_9.txt');
fs = 1/(raw(2,1)-raw(1,1));
yraw = raw(:,2);
t = raw(:,1);
f_o = 0.2*fs;
%create filter
h = exp(-[0:1:length(yraw)-1]*2*pi*(f_o/fs));
%flip filter as convolution is the same as cr-correlation with one signal reversed in time 
fh = flip(h);
%fh = fh;
yraw = yraw';
%apply correlation
y = crcorr(yraw,fh);

%plot result
plot(t,yraw)
hold on
plot(t,y(1:length(t)))
legend('Original Signal','Filtered Signal')
xlabel('time [s]')
ylabel('Amplitude')
figure
freqz(fh)



