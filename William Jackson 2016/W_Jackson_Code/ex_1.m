%Ex1 Format & Purpose
a11t1 = importdata('ex1_1_trend1.txt');
x = 1:length(a11t1);
a11t1 = a11t1.';
plot(a11t1);
%subtract dc offset
a11t1m(x) = mean(a11t1);
a11t1_m = a11t1-mean(a11t1);
hold on
plot(a11t1m);
plot(a11t1_m);
legend('Original Signal','Trend','Corrected Signal')
%%
a11t2 = importdata('ex1_1_trend2.txt');
a11t2 = a11t2.';

figure
plot(a11t2);
%fit first order poly and subtract to remove trend
[p,S] = polyfit(x,a11t2,1);
a11t2_trend = polyval(p,x);
hold on 
plot(a11t2_trend);

a11t2_ = a11t2-a11t2_trend;
plot(a11t2_)
legend('Original Signal','Trend','Corrected Signal')
%%

a11t3 = importdata('ex1_1_trend3.txt');
x = 1:length(a11t3);
a11t3 = a11t3.';

figure
plot(a11t3);
%fit second order poly and subtract to remove trend

[p2,S2] = polyfit(x,a11t3,2);
a11t3_trend = polyval(p2,x);
hold on 
plot(a11t3_trend);

a11t3_ = a11t3-a11t3_trend;
plot(a11t3_)
legend('Original Signal','Trend','Corrected Signal')



