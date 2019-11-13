function r = fn_xcorr(x, y, n)
%SUMMARY
%   Cross-correlation function with or without normalisations
%USAGE
%	r = fn_xcorr(x,y,n)
%AUTHOR
%	Euan Foster (2019)
%OUTPUTS
%	x - 1D array for correlation
%   y - 1D array for correlation
%   n - option for normalisation
%     - 1 for normalisation
%     - 2 for without normalisation
%INPUTS
%	r - cross correlation product with and without normalisation
%NOTES
%	output is a row vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculating the Mean of the signals
x_bar = mean(x);
y_bar = mean(y);

%Calcating max length
x_len = length(x);
y_len = length(y);
len = max(x_len,y_len);

%pad signals for shifting
x(length(x)+1:2*len) = 0;
y(length(y)+1:2*len) = 0;
x = padarray(x,[0 len],0,'both');
y = padarray(y,[0 len],0,'both');

%calculate normalising denominator
den = sqrt( sum( (x-x_bar).^2) * sum ( (y-y_bar).^2) );
ii = 1;

%Initialising Array
r = zeros(1, ((2*len) - 1) );

if n == 1
    %With Normalisation
    for k = -(len-1):len-1
        y_temp = circshift(y(:),k,1)';
        num = sum( x.*y_temp );
        r(ii) = num./den;
        ii = ii+1;
    end

elseif n == 2
    
    %Without Normalisation
    for k = -(len-1):len-1
        y_temp = circshift(y(:),k,1)';
        num = sum( x.*y_temp );
        r(ii) = num;
        ii = ii+1;
    end
    
end