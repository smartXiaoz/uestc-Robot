  %% BP 网络 
%% 输入输出都是高维
clear all; close all;
x = eye(6);
N = length(x);
D = [    0         0         0         0    0.6400         0
         0         0         0    0.8000         0    0.8000
         0         0         0    0.5120         0         0
         0    0.6400    0.6400         0    0.6400         0
    0.8000         0         0    0.8000         0    0.8000
         0    1.0000         0         0    1.0000    1.0000];
y = D;
 % 非线性函数 
n = 6; % input
p = 6; % output
m = 6; % middle
Num = 2000; % 训练次数
W1 = randn(m,n);b1 = randn(m,1); W2 = randn(p,m);
b2 = randn(p,1); % 参数赋值

mu = 0.5;
for ii = 1:Num
    for jj =1:N
        xx = x(:,jj);
        z2 = W1 * xx + b1;
        a2 = 1./(1+exp(-z2));
        z3 = W2 * a2 + b2;
        fx(:,jj) = 1./(1+exp(-z3));
        
        %% NN trainning
        dd = y(:,jj);
        ee = dd -fx(:,jj);
        Jb1 = ee .* fx(:,jj) .* (1-fx(:,jj)); %与低维相比 全部变为点乘
        Jb2 = a2 .* (1-a2); %修改jb2的值
        b2 = b2 + mu * Jb1;
        W2 = W2 + mu * Jb1 * a2';
        b1 = b1 + mu * W2'* Jb1 .* Jb2;%增加W2 点乘
        W1 = W1 + mu * W2' * Jb1 .* Jb2 * xx';%增加W2 点乘
        
    end
    Err(ii) = norm(y-fx) ;
end
% figure,hold on;
% plot(x,y);
% plot(x,fx);
% legend('original','NN');

figure,hold on;
plot(Err);
xlabel('iterations');
ylabel('Error')
% fx = 1./(1+exp(-x));
% figure,plot(x,fx);