import math
import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
from Model.Critic import Critic_LSTM,Critic_Transformer
from Setting import arg
from Model.Env import Env
#加载测试使用代码


class Double_DQN():
    def __init__(self,network=None,target_network=None):
        #模型可以改
        self.BATCH_SIZE = 128
        self.LR = 0.001
        self.GAMMA = 0.99
        self.MEMORY_SIZE = 10000
        self.MEMORY_THRESHOLD = 5000
        self.UPDATE_TIME = 100
        self.ACTIONS_SIZE = 2

        self.network, self.target_network = network,target_network
        self.memory = deque()
        self.learning_count = 0
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()
        self.loss_record=[]
        self.device = torch.device("cpu")
        self.network=self.network.to(device=self.device)
        self.target_network = self.target_network.to(device=self.device)

    def action(self, state,ResistancePointFlag,hold_time,israndom=True):
        if israndom and random.random() < 0.01:
            return np.random.randint(0, self.ACTIONS_SIZE)

        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device=self.device)
        ResistancePointFlag = torch.tensor([[ResistancePointFlag]],dtype=torch.float32).to(device=self.device)
        hold_time = torch.tensor([[hold_time]],dtype=torch.float32).to(device=self.device)
        actions_value = self.network.forward(state,ResistancePointFlag,hold_time).to(device=self.device)
        return torch.max(actions_value,1)[1].cpu().data.numpy()[0]
        #print(actions_value)
        #return torch.max(actions_value,1)[1].data.numpy()[0]

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
        ResistancePointFlag=torch.FloatTensor([[x[5]] for x in batch]).to(device=self.device)
        hlod_time = torch.FloatTensor([[x[6]] for x in batch]).to(device=self.device)

        actions_value = self.network.forward(next_state,ResistancePointFlag,hlod_time)
        next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1)
        eval_q = self.network.forward(state,ResistancePointFlag,hlod_time).gather(1, action)
        next_q = self.target_network.forward(next_state,ResistancePointFlag,hlod_time).gather(1, next_action)
        target_q = reward + self.GAMMA * next_q * done
        loss = self.loss_func(eval_q, target_q)
        self.loss_record.append(loss.cpu().item())
        print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':

    #env = Env(data_path="Data/B000300/RESSET_INDXSH2022_000300.csv")

    state_dim = 8
    hidden_size = state_dim*2


    agent = Double_DQN(state_dim=state_dim,hidden_size=hidden_size)
    epoch=5
    for k in range(epoch):#开的游戏回合
        print(k)
        env = Env(data_path="Data/B000300/RESSET_INDXSH2022_000300.csv")
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_reward = 0
        episode_timesteps = 0
        done = 1
        before_n = 3
        while env.TimeCursor<env.DataLen:
            if done==1:
                if total_timesteps != 0:
                    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                    if env.DataLen-env.TimeCursor<242*20:
                        break


                done = -1#游戏未开始
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1


            bar=env.Data.loc[env.TimeCursor,:]
            #print(bar['time'])
            if done==-1:
                flag=0
                for j in range(before_n):
                    # 接近前n个阻力位任意一个就开仓
                    if abs(math.log(bar['avg'] / env.ResistancePoint[-1][-1])) < 0.001:
                        obs, reward, done=env.step(action=1)
                        done=0
                        flag=1
                        break
                    if abs(math.log(bar['avg'] / env.SupportPoint[-1][-1])) < 0.001:
                        obs, reward, done=env.step(action=1)
                        done=0
                        flag=1
                        break
                if flag==0:
                    obs, reward, done=env.step(action=0)
                continue


            if done==0 :
                obs = env.Observation
                if env.hold_time>180:#游戏开始，开始持仓,持仓大于180分钟
                    obs=obs.values
                    obs=obs.astype(np.float64)
                    action = agent.action(obs,israndom=True)
                    #print(action)
                    new_obs, reward, done = env.step(action)
                    new_obs=new_obs.values
                    new_obs=new_obs.astype(np.float64)
                    episode_reward += reward
                    agent.learn(obs, action, reward, new_obs, done)
                    obs = new_obs
                    episode_timesteps += 1
                    total_timesteps += 1
                    timesteps_since_eval += 1

                else:
                    action=1
                    new_obs, reward, done = env.step(action)
