import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

class ReplayBuffer(object):
    """docstring for ReplayBuffer"""
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = collections.namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        t = self.Transition(*zip(*random.sample(self.memory, batch_size)))
        return t.state, t.action, t.next_state, t.reward, t.done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, action_class, state_shape):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[-1], 16, 8, 4)
        nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        nn.init.xavier_normal(self.conv3.weight)
        self.dense1 = nn.Linear(64*7*7, 512)
        nn.init.xavier_normal(self.dense1.weight)
        self.dense2 = nn.Linear(512, action_class)
        nn.init.xavier_normal(self.dense2.weight)
        #self.conv1 = nn.Sequential(
        #    nn.Conv2d(state_shape[-1], 16, 8, 4),
            #nn.BatchNorm2d(16),
        #    nn.ReLU()
        #)
        #self.conv2 = nn.Sequential(
        #    nn.Conv2d(16, 32, 4, 2),
            #nn.BatchNorm2d(32),
        #    nn.ReLU()
        #)
        #self.conv3 = nn.Sequential(
        #    nn.Conv2d(32, 64, 3, 1),
            #nn.BatchNorm2d(64),
        #    nn.ReLU()
        #)
        #self.dense = nn.Sequential(
        #    nn.Linear(64 * 7 * 7, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, action_class)
        #)        

    def forward(self, x):
        x = x.transpose(1, 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print (x.shape)
        #exit()
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        return self.dense2(x)


class Schedule:
    """docstring for Schedule"""
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.step = 0

    def take_action(self):
        if self.step < 0.3 * self.timestamp:
            ret = (0.3 * self.timestamp - self.step) / (0.3 * self.timestamp)
        else:
            ret = 0.02
        #ret = max( * (self.timestamp - self.step) / self.timestamp, 1)
        #ret = np.exp(-5 * self.step / self.timestamp)
        self.step += 1
        return ret

        

