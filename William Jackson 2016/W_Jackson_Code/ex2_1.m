% NDE of an aerospace component 
close all
clear 
%% import data
raw = importdata('ex2_1.txt');
volt = raw(:,1);
fs = 40e6;
thickness = 4e-3;
%% create time axis % plot
T =  1/fs;
t = (0:T:T*length(volt)-T); 
plot(t,volt);
xlabel('time [s]')
ylabel('Amplitude')
%% segment and window echoes
frontwall = volt(44:94);
backwall = volt(152:202);
%wind = kaiser(length(frontwall),15);
wind = tukeywin(length(frontwall),0.8);
frontwallw = frontwall.*wind;
backwallw = backwall.*wind;

%% plot windowed signals
figure(100)
plot(frontwall/max(frontwall));
hold on 
plot(frontwallw/max(frontwallw));
plot(wind);
ylabel('Amplitude (normalised)')
xlabel('Sample Number')
legend('Original Signal','Windowed Signal','Window Function')
figure(110)
plot(backwall/max(backwall));
hold on 
plot(wind)
plot(backwallw/max(backwallw));
ylabel('Amplitude (normalised)')
xlabel('Sample Number')
legend('Original Signal','Windowed Signal','Window Function')

%% perform fft on signals
frontwallw(length(frontwall)+1:2056) = 0;
backwallw(length(frontwall)+1:2056) = 0;
db = 0;
doplot = 0;
f_freq = fft(frontwallw);
b_freq = fft(backwallw);



if doplot == 1
if db ==0;
f_freq = abs(f_freq);
b_freq = abs(fft(backwallw));
f=1e-6*(0:fs/length(f_freq):fs-fs/length(f_freq));
figure(200)
plot( f(1:end/2+1),f_freq(1:end/2+1) )
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')
ylim([min(min(f_freq,b_freq)) max(max(f_freq,b_freq))])
figure(210)
plot(f(1:end/2+1),b_freq(1:end/2+1))
xlabel('Frequency [MHz]')
ylabel('Amplitude [V]')
ylim([min(min(f_freq,b_freq)) max(max(f_freq,b_freq))])
else
f_freq = abs(fft(frontwallw));
f_freq = f_freq/max(f_freq);
f_freq = mag2db(f_freq);
b_freq = abs(fft(backwallw));
b_freq = b_freq/max(b_freq);
b_freq = mag2db(b_freq);
f=1e-6*(0:fs/length(f_freq):fs-fs/length(f_freq));
figure(200)
plot(f(1:end/2+1),f_freq(1:end/2+1))
xlabel('Frequency [MHz]')
ylabel('Amplitude [dB]')
ylim([min(min(f_freq,b_freq)) 0])
figure(210)
plot(f(1:end/2+1),b_freq(1:end/2+1))
xlabel('Frequency [MHz]')
ylabel('Amplitude [dB]')
ylim([min(min(f_freq,b_freq)) 0])
end
end

%% Analysis of fft
f_freqa = (fft(frontwallw));

b_freqa = (fft(backwallw));
f=1e-6*(0:fs/length(f_freq):fs-fs/length(f_freq));


%over 4 mm 
attenuation = f_freqa./b_freqa;
%per meter 
attenuation = attenuation /thickness;
attenuation = abs(attenuation);
attenuation = attenuation/max(attenuation);
attenuation = mag2db(attenuation);

%single sided  
attenuatation = attenuation(1:end/2+1);
figure(300)
plot(f(1:end/2+1),attenuatation)
xlabel('Frequency [MHz]')
ylabel('Attenuation [Nepers.m^{-1}]')
