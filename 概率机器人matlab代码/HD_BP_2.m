  %% BP ���� 
%% ����������Ǹ�ά
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
 % �����Ժ��� 
n = 6; % input
p = 6; % output
m = 6; % middle
Num = 2000; % ѵ������
W1 = randn(m,n);b1 = randn(m,1); W2 = randn(p,m);
b2 = randn(p,1); % ������ֵ

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
        Jb1 = ee .* fx(:,jj) .* (1-fx(:,jj)); %���ά��� ȫ����Ϊ���
        Jb2 = a2 .* (1-a2); %�޸�jb2��ֵ
        b2 = b2 + mu * Jb1;
        W2 = W2 + mu * Jb1 * a2';
        b1 = b1 + mu * W2'* Jb1 .* Jb2;%����W2 ���
        W1 = W1 + mu * W2' * Jb1 .* Jb2 * xx';%����W2 ���
        
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