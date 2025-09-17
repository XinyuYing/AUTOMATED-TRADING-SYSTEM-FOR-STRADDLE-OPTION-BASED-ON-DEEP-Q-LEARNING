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
    def __init__(self,state_dim=10,
                 hidden_size=10,
                 ModelType='lstm',
                 BATCH_SIZE = 128,
                 MEMORY_THRESHOLD=5000
                 ):
        #模型可以改
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = 0.01
        self.GAMMA = 0.99
        self.MEMORY_SIZE = 15000
        self.MEMORY_THRESHOLD = MEMORY_THRESHOLD
        self.UPDATE_TIME = 100
        self.ACTIONS_SIZE = 3

        if ModelType=='lstm':
            self.network, self.target_network = Critic_LSTM(state_dim,hidden_size), Critic_LSTM(state_dim,hidden_size)
        else:
            self.network, self.target_network = Critic_Transformer(state_dim, hidden_size), Critic_Transformer(state_dim, hidden_size)

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

    def learn(self, state, action, reward, next_state, done,ResistancePointFlag,hold_time):
        if done==1:
            self.memory.append((state, action, reward, next_state, 0,ResistancePointFlag,hold_time))
        else:
            self.memory.append((state, action, reward, next_state, 1,ResistancePointFlag,hold_time))
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.popleft()
        if len(self.memory) < self.MEMORY_THRESHOLD:
            return

        if self.learning_count % self.UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.learning_count += 1

        batch = random.sample(self.memory, self.BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch]).to(device=self.device)
        action = torch.LongTensor([[x[1]] for x in batch]).to(device=self.device)
        reward = torch.FloatTensor([[x[2]] for x in batch]).to(device=self.device)
        next_state = torch.FloatTensor([x[3] for x in batch]).to(device=self.device)
        done = torch.FloatTensor([[x[4]] for x in batch]).to(device=self.device)


        actions_value = self.network.forward(next_state)
        next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1)
        eval_q = self.network.forward(state).gather(1, action)
        next_q = self.target_network.forward(next_state).gather(1, next_action)
        target_q = reward + self.GAMMA * next_q * done
        loss = self.loss_func(eval_q, target_q)
        self.loss_record.append(loss.cpu().item())

        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


