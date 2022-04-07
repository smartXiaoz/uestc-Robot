clear all;
close all;
clc

x = -10:0.1:10;
noise = randn(size(x))*3;
y = 3 * x + 1 + noise;
A = [x', ones(size(x))'];

y = 3*x.^3 + 2 * x.^2 + 3.*x + 1 + noise;
A = [x.^3',x.^2', x', ones(size(x))'];
a = inv(A'*A)*A'*y';

figure 
hold on
plot(x,y,'.')
plot(x,A*a)
