function[r_xy] = crcorr(x,y)
if nargin < 2
   y = x;
end
%calculate means of signals
menx = mean(x);
meny = mean(y);
%calculate MATLAB result
matcor = xcorr(x,y);
%get lengths of signals
x_len = length(x);
y_len = length(y);
%select max length
len = max(x_len,y_len);
%pad signals for shifting
x(x_len+1:2*len) = 0;
y(y_len+1:2*len) = 0;
x = padarray(x,[0 len],0,'both');
y = padarray(y,[0 len],0,'both');

%calculate normalising denominator
r_xy_den = sqrt(    sum( (x-menx).^2) * sum ( (y-meny).^2));
ii = 1;

%shift signal y signal across x, calculate pointwise multiplication at each
%index & summate
for k = -(len-1):len-1
    y_circ = circshift(y(:),k,1)';
    inner = (x).*(y_circ);
    %inner = (x-menx)).*(y_circ-meny);
    r_xy_num(ii) = sum(inner);
    r_xy(ii) = r_xy_num(ii);%./r_xy_den;
%     plot(x)
%     hold on
%     plot(y_circ)
%     plot(r_xy)
%     hold off
%     pause(0.3)
%     legend('X','Y','Correlation')
%     ylabel('Amplitude')
%     xlabel('Index')
     ii = ii+1;
end
% %plot
% stem(x,'-b')
% hold on
% stem(y,'-g')
% stem(r_xy,'xk')
% stem(matcor,'--r')
% legend('x','y','r\_xy','mat cor')
% legend('X','Y','Correlation','MATLAB Correlation')
% ylabel('Amplitude')
% xlabel('Index')

