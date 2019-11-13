%ex 114 digital filter design
close all 
clear 
raw = importdata('ex1_9.txt');
fs = 1/(raw(2,1)-raw(1,1));
x = raw(:,2)';
N = 2; % max z coefficient
t = raw(:,1);
%target freq = 20MHz 
p = 0.90;
thet = 2*pi*(2e7/fs);
a(1) = -2*cos(thet);
b(1) = 2*p*cos(thet);
b(2) = -p^2;
%uncomment for impulse
%  x(1:length(x)) = 0;
%  x(3) = 1;

ymn(1:length(x)) = 0;
m = (N+1:length(x));
%band pass
for mm = m(1:end)
    
ymn(mm) = b(1)*ymn(mm-1) + b(2)*ymn(mm-2) +x(mm) +a(1)*x(mm-1) + x(mm-2);

end
%y = -y(m-2) +x(m) -x(m-4);

fnot = abs(fft(ymn(1:256)));
fnot = fnot/max(fnot);
fnot = mag2db(fnot);
xfft = abs(fft(x(1:256)));
xfft = xfft/max(xfft);
xfft = mag2db(xfft);
f=(0:fs/length(fnot):fs-fs/length(fnot));
figure
plot(f(1:end/2+1),xfft(1:end/2+1))

hold on
plot(f(1:end/2+1),fnot(1:end/2+1))

xlabel('Frequency [Hz]')
ylabel('Amplitude [dB]')
legend('Original Signal','Notch Filtered')
ylim([-50 0])
xlim([0 40e6]);
figure
plot(t,x);
hold on
plot(t,ymn);
xlabel('time [s]')
ylabel('Amplitude')
legend('Original Signal','Notch Filtered')
