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
network=Critic_Transformer(state_dim=state_dim,hidden_size=hidden_size,droupout=0.1)
target_network=Critic_Transformer(state_dim=state_dim,hidden_size=hidden_size,droupout=0.1)
ModelType='lstm'
network.load_state_dict(torch.load('ModelParam/2024-3-14/64network-TransformerNoguiadance.pth'))
target_network.load_state_dict(torch.load('ModelParam/2024-3-14/64target-network-TransformerNoguiadance.pth'))
performance=pd.DataFrame(columns=['ProfitRate','Time'])
agent = Double_DQN(network=network,target_network=target_network)
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
    while env.TimeCursor < env.DataLen:
        if env.DataLen - env.TimeCursor < arg.ADayTime * 2:
            break
        new_data=dict()
        bar=env.Data.loc[env.TimeCursor,:]
        Assert=env.account.AllCash+env.account.getMarketValue(price=bar['close'],time=bar['time'],IV=env.HV)
        new_data['ProfitRate'] = Assert / env.account.initCash
        new_data['Time'] = bar['time']
        new_Data=pd.DataFrame(new_data,index=[0])
        performance=pd.concat([performance,new_Data],ignore_index=True,axis=0)

        if done == 1:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
                # if env.DataLen - env.TimeCursor < arg.ADayTime*5:
                #     break

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
                action = agent.action(obs,ResistancePointFlag=env.ResistancePointFlag,hold_time=env.hold_time,israndom=True)

            if env.hold_time>112 or env.Observation.loc[env.Observation.shape[0]-1,'NextDay']>3:
                action=0

            new_obs, reward, done, = env.step(action)
            new_obs = new_obs.values
            new_obs = new_obs.astype(np.float64)
            episode_reward += reward
            #agent.learn(obs, action, reward, new_obs, done,env.ResistancePointFlag,env.hold_time)
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1



env.HistoricalTransaction.to_csv('HistoricalTransaction.csv',index=False)
performance.to_csv('Performance.csv',index=False)