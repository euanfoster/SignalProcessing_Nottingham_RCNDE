%Ex1_2
close all 
clear
N = 100;
M = 1000;
for ii =  1:M
    X(ii,:) = rand(1,N);
end
X_sd = std(X);
X_sd_mean = mean(X_sd);
X_c = sum(X)/M;
X_c_mean = std(X_c);
fprintf('Expected Value: %f \nActual Value: %f\n',sqrt(M),X_sd_mean/X_c_mean)

%Expected Value: 31.622777
%Actual Value: 28.970080
