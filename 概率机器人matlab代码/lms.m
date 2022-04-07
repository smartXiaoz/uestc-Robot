clear all; close all;
L = 2000;

%X = [ [-1 1 -1 1]', [ 1 1 -1 1]' ];
 
X = -1 : 0.1 : 1;
D = sin(X);
D = D / 6 + 0.5;
N = size(D, 2); m = 5;
n = size(X, 1); 
w1 = zeros(m,n);
b1 = ones(m, 1);
w2 = randn(1, m);
b2 = randn(1);
mu = 8;
for kk = 1 : L
    for ii = 1 : N
       x = X(:,ii);
       z2 = w1 * x + b1;
       a2 = 1 / (1 + exp(-z2));
       z3 = w2 * a2' + b2;
       y = 1 / (1 + exp(-z3));
       e = D(:,ii) - y;
       tmp1 = e * y .* (1-y);
       tmp2 = a2 .* (1 - a2);
       w2 = w2 + mu * tmp1 * a2;
       b2 = b2 + mu * tmp1;
       w1 = w1 + mu * tmp1 * w2' .* tmp2' .* x';
       b1 = b1 + mu * tmp1 * w2' .* tmp2';
       
end
    Err(kk) = norm(e);
end
 
for ii = 1:N
   x = X(:,ii);
   z2 = w1 * x + b1;
   a2 = 1 / (1 + exp(-z2));
   z3 = w2 * a2' + b2;
   y = 1 / (1 + exp(-z3));
   fz(ii) = y;
end
 
figure,plot(Err);
figure, hold on;
plot(D,'ro'); plot(fz,'g*');
legend('REAL', 'TRAIN');

 
