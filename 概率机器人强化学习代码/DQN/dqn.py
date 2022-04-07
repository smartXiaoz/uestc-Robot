import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 500000
batch_size    = 256


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_prime = s_prime.to(device)
        done_mask = done_mask.to(device)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def main():
    env = gym.make('CartPole-v1')
    q = Qnet().to(device)
    # q.load_state_dict(torch.load(r'./model/model1900.pth'))
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    returns = []
    for n_epi in range(1000000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
        if memory.size() > 2000:

            train(q,q_target,memory,optimizer)


        PATH = './model'
        if n_epi % 10 == 0:
            returns.append(score/print_interval)
            plt.figure()
            plt.plot(range(len(returns)), returns)
            plt.savefig(PATH + '/plt.png', format='png')
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))

            PATH1 = './model/model' + str(n_epi) + '.pth'
            if score / print_interval > 50:
                torch.save(q.state_dict(), PATH1)
                print('save')


            score = 0.0
    env.close()


def test():
    env = gym.make('CartPole-v1')
    mu = Qnet()
    mu.load_state_dict(torch.load(r'./model/model46980.pth'))
    for i in range(100):
        s = env.reset()
        done = False
        while not done:
            a = mu.sample_action(torch.from_numpy(s).float(), 0)
            # a = a.item()
            print(a)
            s_prime, r, done, info = env.step(a)
            s = s_prime
            env.render()
        print('done')
    env.close()
if __name__ == '__main__':

    main()
    # test()