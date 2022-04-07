import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 超参数
lr_mu = 0.0005 #动作网络学习率
lr_q = 0.001 #评价网络学习率
gamma = 0.99 #折扣回报率
batch_size = 32
buffer_limit = 50000
tau = 0.005 #target软更新比例

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition): # 状态 动作 奖励 下一时刻状态 done
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch: # s [32, 3] a [32, 1]
            s, a, r, s_prime, done_mast = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mast = 0.0 if done_mast else 1.0
            done_mask_lst.append([done_mast])
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float),\
               torch.tensor(r_lst, dtype=torch.float),torch.tensor(s_prime_lst, dtype=torch.float),\
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class MuNet(nn.Module): # 动作网络
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        out = torch.tanh(x) * 2 # 乘2是因为动作空间-2，2
        return out

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def train(mu, mu_target, Q, Q_target, memory, Q_optim, mu_optim):
    s, a, r, s_prime, done = memory.sample(batch_size)

    target = r + gamma * Q_target(s_prime, mu_target(s_prime)) * done # 当前奖励， 下一时刻选择动作，Q值
    q_loss = F.smooth_l1_loss(Q(s, a), target.detach())
    Q_optim.zero_grad()
    q_loss.backward()
    Q_optim.step() #完成 critic网络更新

    mu_loss = -Q(s, mu(s)).mean() #梯度上升
    mu_optim.zero_grad()
    mu_loss.backward()
    mu_optim.step()

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def main():
    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer()

    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())
    Q, Q_target = QNet(), QNet()
    Q_target.load_state_dict(Q.state_dict())

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    Q_optimizer = optim.Adam(Q.parameters(), lr=lr_q)

    score = 0.0
    print_interval = 20
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(1000):
        s = env.reset()
        done = False

        while not done:
            a = mu(torch.from_numpy(s).float()) #直接使用网络采样
            a = a.item() + ou_noise()[0]
            s_prime, r, done, info = env.step([a])
            memory.put((s, a, r/100.0, s_prime, done))
            score += r
            s = s_prime

        if memory.size() > 2000:
            for i in range(10):
                train(mu,mu_target,Q,Q_target,memory,Q_optimizer,mu_optimizer)
                soft_update(mu,mu_target)
                soft_update(Q,Q_target)

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            PATH = './model/model_' + str(n_epi) + '.pth'
            if score / print_interval > -500:
                torch.save(mu.state_dict(), PATH)
                print('save')
            score = 0.0

    env.close()

def test():
    env = gym.make('Pendulum-v0')
    mu = MuNet()
    mu.load_state_dict(torch.load(r'./model/model_980.pth'))

    done = False
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    for i in range(1000):
        s = env.reset()
        print(i)
        done = False
        while not done:
            a = mu(torch.from_numpy(s).float())
            a = a.item() + ou_noise()[0]
            # print(a)
            s_prime, r, done, info = env.step([a])
            s = s_prime
            env.render()
            print(done)
    env.close()
if __name__ == '__main__':
    # main()
    test()
