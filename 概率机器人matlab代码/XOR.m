clear all; close all;
L = 2000;
N = 4; m = 4;
%X = [ [-1 1 -1 1]', [ 1 1 -1 1]' ];
X  = [ -1 1  1 -1
         1 1   1 -1
         -1 -1  1 -1
         1 1 1 1 ];
D =[1 0 0.5 0.2];
 
 
% X  = [ 1 0 1 0
%          0 1 1 0
%          1 1 1 1];
% D = [ 1 1 0 0];
% D = [ 0 0 1 1];
 
% D = [ 0 0 0 0];%ok
% D = [ 0 0 0 1];%ok
% D = [ 0 0 1 0];%ok
D = [ 0 0 1 1];% NO NO NO
% D = [ 0 1 0 0];%ok
% D = [ 0 1 0 1];%ok
% D = [ 0 1 1 0];%ok
% D = [ 0 1 1 1];%ok
% D = [ 1 0 0 0];%ok
% D = [ 1 0 0 1];%ok
% D = [ 1 0 1 0];%ok
% D = [ 1 0 1 1];%ok
% D = [ 1 1 0 0]; %% NO NO NO
% D = [ 1 1 0 1];%ok
% D = [ 1 1 1 0];%ok
D = [ 1 1 1 1]; %ok
 
w = randn(m,1);
mu = 8;
for kk = 1 : L
    for ii = 1 : N
        x = X(:,ii);
        z = w' * x;
        fz = 1 / (1+exp(-z));
        d = D(ii);
        e = d - fz;
        w = w + mu * e * fz * (1-fz) * x;
    end
    Err(kk) = norm(e);
end
 
for ii = 1:N
    x = X(:,ii);
    z = w' * x;
    fz(ii) = 1 / (1+exp(-z));
end
 
figure,plot(Err);
figure, hold on;
plot(D,'ro'); plot(fz,'g*');
legend('REAL', 'TRAIN');
axis([-0.1 4.1 -0.1 1.1])

