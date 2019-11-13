%ex 1_3
close all
clear
phase = 0;
cycles = 20;
%create 1024 x values for two cycles
x  = linspace(0,4*pi,1024);
%create 1024 x values for double input cycles
x2 = linspace(0,2*cycles*pi,1024);
%calculate cosine
y = cos(x);
%calculate cosine with phase change
y2 = cos(x2+phase);
y2 = sin(x2+phase);
%pointwise multiply signals
y_y2 = y.*y2;
%calculate mean of pointwise multiplied signals (integral)
int_y_y2 = mean(y_y2);
plot(x,y,x,y2,x,y_y2);
string = strcat('Sin(','x','+',num2str(phase),')');
legend('Cos(x)',string,'Product');
xlabel('Angle [rad]')
ylabel('Amplitude')
annotation('textbox','String',['Integral:', num2str(int_y_y2)],'FitBoxToText','on');
r_xy_num = sum( (y-mean(y)).*(y2-mean(y2))  );
r_xy_den =sqrt(sum( (y-mean(y)).^2) * sum ((y2-mean(y2)).^2));
r_xy = r_xy_num/r_xy_den;




