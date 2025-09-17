import math
import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
from Model.Critic import Critic_LSTM,Critic_Transformer
from Setting import arg
from Model.Env import Env



class Double_DQN():
    def __init__(self,
                 network=None,
                 target_network=None
                 ):
        #模型可以改

        self.LR = 0.01
        self.GAMMA = 0.99
        self.MEMORY_SIZE = 15000

        self.UPDATE_TIME = 100
        self.ACTIONS_SIZE = 3


        self.network, self.target_network = network, target_network


        self.memory = deque()
        self.learning_count = 0
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()
        self.loss_record = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network=self.network.to(device=self.device)
        self.target_network = self.target_network.to(device=self.device)

    def action(self, state, israndom):


        if israndom and random.random() < 0.1:
            return np.random.randint(0, self.ACTIONS_SIZE)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device=self.device)

        actions_value = self.network.forward(state).to(device=self.device)
        return torch.max(actions_value,1)[1].cpu().data.numpy()[0]




