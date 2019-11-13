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

y = zeros(1,2*N-1);

for ii = 1:N
    h = exp(-((ii-1):-1:0).*2*pi*(F0/Fs));
    y(ii) = voltage(1:ii)*h';
end

figure(01)
plot(time,y(1:N),'b',time,voltage,'--g','Linewidth',2.0)
