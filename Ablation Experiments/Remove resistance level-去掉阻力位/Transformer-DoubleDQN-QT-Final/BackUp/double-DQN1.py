#间隔一段时间结算一次

import math
import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
from Model.Critic import Critic_LSTM,Critic_Transformer
from Setting import arg
from Model.Env import Env


BATCH_SIZE = 32
LR = 0.001
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
TOTAL_EPISODES = 10000000
MEMORY_SIZE = 1000000
MEMORY_THRESHOLD = 100
UPDATE_TIME = 10000
TEST_FREQUENCY = 1000
ACTIONS_SIZE=2




class Agent(object):
    def __init__(self,state_dim=8,hidden_size=8,num_layer=1):
        self.network, self.target_network = Critic_LSTM(state_dim,hidden_size,num_layer), Critic_LSTM(state_dim,hidden_size,num_layer)
        self.memory = deque()
        self.learning_count = 0
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def action(self, state, israndom):
        if israndom and random.random() < 0.5:
            return 1
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.network.forward(state)
        return torch.max(actions_value,1)[1].data.numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        if done==1:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()
        if len(self.memory) < MEMORY_THRESHOLD:
            return

        if self.learning_count % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.learning_count += 1

        batch = random.sample(self.memory, BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])

        actions_value = self.network.forward(next_state)
        next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1)
        eval_q = self.network.forward(state).gather(1, action)
        next_q = self.target_network.forward(next_state).gather(1, next_action)
        target_q = reward + GAMMA * next_q * done
        loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':

    #env = Env(data_path="Data/B000300/RESSET_INDXSH2022_000300.csv")

    state_dim = 8
    hidden_size = state_dim*2


    agent = Agent(state_dim=state_dim,hidden_size=hidden_size)
    epoch=5
    for k in range(epoch):#开的游戏回合
        print(k)
        env = Env(data_path="Data/B000300/RESSET_INDXSH2022_000300.csv")
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_reward = 0
        episode_timesteps = 0
        done = -1
        before_n = 3
        performance_time=242*20#业绩考核日
        while env.TimeCursor<env.DataLen:

            performance_time=performance_time-1
            print(env.TimeCursor)
            print(performance_time)

            if done==1:
                print(episode_reward)
                done = -1#游戏未开始

            bar=env.Data.loc[env.TimeCursor,:]
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
                    new_obs, reward, done = env.step(action)
                    new_obs=new_obs.values
                    new_obs=new_obs.astype(np.float64)
                    episode_reward += reward
                    if done==1 and performance_time<=0:#间隔20个交易日，回合终止

                        agent.learn(obs, action, reward, new_obs, done)

                        print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                        total_timesteps, episode_num, episode_timesteps, episode_reward))

                        performance_time=242*20
                        episode_reward = 0
                        episode_timesteps = 0
                        episode_num += 1

                    elif done==1 and env.DataLen-env.TimeCursor<242*20:
                        agent.learn(obs, action, reward, new_obs, done)
                        break

                    else:
                        agent.learn(obs, action, reward, new_obs, 0)#不到20个交易日，继续
                    obs = new_obs
                    episode_timesteps += 1
                    total_timesteps += 1
                    timesteps_since_eval += 1

                else:
                    action=1
                    new_obs, reward, done = env.step(action)
