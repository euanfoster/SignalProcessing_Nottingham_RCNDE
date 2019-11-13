%ex 114 digital filter design
close all 
clear 
raw = importdata('ex1_9.txt');
fs = 1/(raw(2,1)-raw(1,1));
x = raw(:,2)';
t = raw(:,1);
N = 4;
fs = 1/(raw(2,1)-raw(1,1));
ymb(1:length(x)) = 0;

ymh(1:length(x)) = 0;
m = (N+1:length(x));
%band pass
for mm = m(1:end-N)
    
ymb(mm) = -ymb(mm-2) +x(mm) - x(mm-4);

end
for mm = m(1:end-N)
    
ymh(mm) = -ymh(mm-1) +x(mm) - x(mm-4);

end
%y = -y(m-2) +x(m) -x(m-4);

plot(t,x)
hold on 
plot(t,ymb)
plot(t,ymh)
xlabel('time [s]')
ylabel('Amplitude')

legend('Original Signal','Band Pass','High Pass')


