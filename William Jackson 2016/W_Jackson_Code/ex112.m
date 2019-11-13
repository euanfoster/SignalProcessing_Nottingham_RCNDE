%ex 112 digital filter design
close all 
clear 
raw = importdata('ex1_9.txt');
fs = 1/(raw(2,1)-raw(1,1));

yraw = raw(:,2);
t = raw(:,1);
f_o = 0.2*fs;

c = fs/(2*pi*f_o); %define c value based on  desired cutoff
x = yraw';
y(1) = c; %set initial y value to c for filter minimal step change
y(2:length(x)) = 0;

m = (2:length(x)); %start from two to remain within signal index
b(1) = 1; %coefficients of filter
b(2) = 1;
a(1) = (1+c);
a(2) = (-c+1);
m_filt = filter(b,a,x); %apply filter
for mm = m(1:end)     
y(mm) = x(mm) + x(mm-1) + y(mm-1)*(c-1);
end
%y = x(m) + x(m-1) + y(m-1)*(c-1);
plot(t,y)
hold on 
plot(t,yraw);
%hold on 
%plot(t,m_filt,'b--');
legend('Original Signal','Filtered Signal')
xlabel('time [s]')
ylabel('Amplitude')
figure
zplane(b,a);
figure
freqz(b,a,1024,fs);

fnot = abs(fft(y(1:256)));
fnot = fnot/max(fnot);
fnot = mag2db(fnot);
xfft = abs(fft(yraw(1:256)));
xfft = xfft/max(xfft);
xfft = mag2db(xfft);
f=(0:fs/length(fnot):fs-fs/length(fnot));
figure
plot(f(1:end/2),xfft(1:end/2))

hold on
plot(f(1:end/2),fnot(1:end/2))

xlabel('Frequency [Hz]')
ylabel('Amplitude [dB]')
legend('Original Signal','Filtered')
