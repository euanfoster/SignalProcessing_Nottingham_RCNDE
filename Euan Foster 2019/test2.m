%% exc 1_6
close all
clear all

%CORRELATION

%Initialising Arrays
x_t = zeros(1,1000);
x_t(1,672:772) = 1;
x_t(1,291:391) = 1;
x = 0:length(x_t) - 1;

y_t = zeros(1,1000);
y_t(1,1:100) = 1;
y = 0:length(y_t) - 1;

x_corr_norm = fn_xcorr(x_t,y_t,1);
x_corr = fn_xcorr(x_t,y_t,2);
x2 = -max(x):max(x);
%check with MATLAB function
[r_check,lags] = xcorr(x_t,y_t);

%Plotting cross correlations results
figure(01)
subplot(5,1,1)
plot(x,x_t);
title('x(t)')
ylabel('Amplitude')
xlabel('Time Samples')
subplot(5,1,2)
plot(y,y_t);
title('y(t)')
ylabel('Amplitude')
xlabel('Time Samples')
subplot(5,1,3)
plot(x2,x_corr_norm);
title('With Normalisation')
ylabel('Amplitude')
xlabel('Lag')
subplot(5,1,4)
plot(x2,x_corr);
title('Without Normalisation')
ylabel('Amplitude')
xlabel('Lag')
subplot(5,1,5)
plot(lags,r_check)
title('Matlab Xcorr Function')
ylabel('Amplitude')
xlabel('Lag')

%AUTO-CORRELATION
y_t = x_t;
auto_corr_norm = fn_xcorr(x_t,y_t,1);
auto_corr = fn_xcorr(x_t,y_t,2);
%Check with MATLAB function
[r_check,lags] = xcorr(x_t,x_t);

figure(02)
subplot(5,1,1)
plot(x,x_t);
title('x(t)')
ylabel('Amplitude')
xlabel('Time Samples')
subplot(5,1,2)
plot(y,y_t);
title('y(t)')
ylabel('Amplitude')
xlabel('Time Samples')
subplot(5,1,3)
plot(x2,auto_corr_norm);
title('With Normalisation')
ylabel('Amplitude')
xlabel('Lag')
subplot(5,1,4)
plot(x2,auto_corr);
title('Without Normalisation')
ylabel('Amplitude')
xlabel('Lag')
subplot(5,1,5)
plot(lags,r_check)
title('Matlab Auto Corr Function')
ylabel('Amplitude')
xlabel('Lag')