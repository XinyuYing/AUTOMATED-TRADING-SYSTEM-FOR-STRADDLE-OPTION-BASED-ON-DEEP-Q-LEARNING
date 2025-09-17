import numpy as np
import torch
from Model.Critic import Critic_LSTM
from Model.Env import Env
from Model.Double_DQN_Load import Double_DQN
from Setting import arg
import pandas as pd

state_dim = 10
hidden_size = 64
network=Critic_LSTM(state_dim=state_dim,hidden_size=hidden_size)
target_network=Critic_LSTM(state_dim=state_dim,hidden_size=hidden_size)
ModelType='lstm'
network.load_state_dict(torch.load('ModelParm/2024年2月10日 18-21年训练，22年测试/64network-lstmNoguiadance.pth'))
target_network.load_state_dict(torch.load('ModelParm/2024年2月10日 18-21年训练，22年测试/64target-network-lstmNoguiadance.pth'))
performance=pd.DataFrame(columns=['ProfitRate','Time'])
agent = Double_DQN(network=network,target_network=target_network)
epoch = 1
for k in range(epoch):  # 开的游戏回合
    #print(k)
    env = Env(data_path="Data/15m000300/RESSET_INDXSH2022_000300.csv")

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = 1
    while env.TimeCursor < env.DataLen:
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
                if env.DataLen - env.TimeCursor < arg.ADayTime * 2:
                    break

            done = 0  # 游戏未开始
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        bar = env.Data.loc[env.TimeCursor, :]

        if done == 0:
            obs = env.Observation
            obs = obs.values
            obs = obs.astype(np.float64)
            action = agent.action(obs,ResistancePointFlag=env.ResistancePointFlag,hold_time=env.hold_time)

            new_obs, reward, done, = env.step(action)
            new_obs = new_obs.values
            new_obs = new_obs.astype(np.float64)
            episode_reward += reward
            #agent.learn(obs, action, reward, new_obs, done,env.ResistancePointFlag,env.hold_time)
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

#torch.save(agent.network.state_dict(),'ModelParam/2024-02-03'+str(hidden_size)+'network-'+ModelType+'.pth')
#torch.save(agent.target_network.state_dict(),'ModelParam/2024-02-03'+str(hidden_size)+'target-network-'+ModelType+'.pth')
#np.save(file='LossRecord'+ModelType+'.npy',arr=np.array(agent.loss_record))

#env.HistoricalTransaction.to_csv('HistoricalTransaction.csv',index=False)