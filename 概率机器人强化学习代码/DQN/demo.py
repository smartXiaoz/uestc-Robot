import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    for i in range(20):
        s = env.reset()
        for j in range(100):
            env.render()
            a = env.action_space.sample()
            env.step(a)

    env.close()


