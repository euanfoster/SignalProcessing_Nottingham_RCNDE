%ex 113 digital filter design

close all 
clear 
raw = importdata('ex1_9.txt');
t = raw(:,1);
fs = 1/(raw(2,1)-raw(1,1));
x = raw(:,2)';
N = 2;
%matlab smoothing filter of N
y_smooth = smooth(x,N,'moving')';
y(1) = 0;
y(N:length(x)) = 0;
m = (N+1:length(x));

for mm = m(1:end)
    
y(mm) = y(mm-1) + x(mm) -x(mm-N);

end

%y = x(m) - x(m-N) + y(m-1);
plot(t,x);
hold on
plot(t,y)
xlabel('time [s]')
ylabel('Amplitude')

legend('Original Signal','Moving Average')


