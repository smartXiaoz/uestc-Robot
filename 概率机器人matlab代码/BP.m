
clear all; close all;
X = -1: 0.1:1;
% X = -2: 0.1:2;
n = size(X,1);
L = size(X,2);
D = sin(4*X) + cos(3*X+rand(1,L)*0.0) + tanh(6 *X);
% D = sin(4*X) + cos(3*X+rand(1,L)*0.0) + tanh(0 *X);
D = D/6 +0.5;
 
Num = 2000;
mu = 0.2; mu2=1; mu3=2; mu4 =3;
m =15;
%% Initial parameters
W1 = randn(m,n); b1 = randn(m,1);
W2 = randn(1,m); b2 = randn(1);
 
%% BP Training
for kk = 1:Num
    for ii=1:L
        x = X(:,ii);
        z2 = W1 * x + b1;
        a2 = 1 ./ (1+exp(-z2));
        z3 = W2 * a2 + b2;
        y(ii) = 1 ./ (1+exp(-z3));
        d = D(ii);
        e = d - y(ii);
        Jb1 = e * y(ii) * (1-y(ii));
        Jb2 = a2 .* (1- a2);
        b2 = b2 + mu4 * Jb1;
        W2 = W2 + mu3 * Jb1 * a2';
        b1 = b1 + mu2 * Jb1 * W2' .* Jb2;
        W1 = W1 + mu * Jb1 * W2' .* Jb2 * x';
        
    end
    Err(kk) = norm(y-D);
end
%% output
for ii=1:L
    x = X(:,ii);
    z2 = W1 * x + b1;
    a2 = 1 ./ (1+exp(-z2));
    z3 = W2 * a2 + b2;
    y(ii) = 1 ./ (1+exp(-z3));       
end
norm(D-y)
X1=-5:0.1:5;
 figure, plot(X1,1 ./ (1+exp(-X1)))
figure, plot(Err)
figure,hold on;
plot(X,D,'r');plot(X,y,'g');
legend('Init','BP');
