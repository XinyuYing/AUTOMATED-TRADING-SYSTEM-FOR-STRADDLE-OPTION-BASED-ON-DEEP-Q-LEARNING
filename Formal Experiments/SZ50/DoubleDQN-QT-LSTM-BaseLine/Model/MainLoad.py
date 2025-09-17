import math
import numpy as np
import torch
from Model.Critic import Critic_LSTM,Critic_Transformer
from Model.Env import Env
from Model.Double_DQN_Load import Double_DQN
from Setting import arg
from BackUp.Guidance import Guidance
import pandas as pd

state_dim = 10
hidden_size = 64
network=Critic_LSTM(state_dim=state_dim,hidden_size=hidden_size)
target_network=Critic_LSTM(state_dim=state_dim,hidden_size=hidden_size)
ModelType='lstm'
network.load_state_dict(torch.load('64network-lstmNoguiadance.pth'))
target_network.load_state_dict(torch.load('64target-network-lstmNoguiadance.pth'))
agent = Double_DQN(network=network,target_network=target_network)
performance=pd.DataFrame(columns=['ProfitRate','Time'])
epoch = 1
for k in range(epoch):  # 开的游戏回合
    #print(k)
    env = Env(data_path="Data/15m000300/RESSET_INDXSH2022_000300.csv")
    guidance=Guidance(env)
    guidanceFlag=0
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = 1
    while env.TimeCursor < env.DataLen-arg.ADayTime*2:

        new_data=dict()
        bar=env.Data.loc[env.TimeCursor,:]
        Assert=env.account.getMarketValue(price=bar['close'])
        new_data['ProfitRate'] = Assert / env.account.initCash
        new_data['Time'] = bar['time']
        new_Data=pd.DataFrame(new_data,index=[0])
        performance=pd.concat([performance,new_Data],ignore_index=True,axis=0)

        if done == 1:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))


            done = 0  # 游戏未开始
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        bar = env.Data.loc[env.TimeCursor, :]

        if done == 0:
            obs = env.Observation
            obs = obs.values
            obs = obs.astype(np.float64)
            if guidanceFlag==1:
                action=guidance.getGuidanceAction()
            else:
                action = agent.action(obs,israndom=True)


            new_obs, reward, done, = env.step(action-1)
            new_obs = new_obs.values
            new_obs = new_obs.astype(np.float64)
            episode_reward += reward

            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

performance.to_csv('Performance.csv',index=False)