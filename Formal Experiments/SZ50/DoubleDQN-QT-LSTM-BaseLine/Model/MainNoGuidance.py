import math
import numpy as np
import torch

from Model.Env import Env
from Model.Double_DQN import Double_DQN
from Setting import arg
from BackUp.Guidance import Guidance
import math
state_dim = 10
hidden_size = 64
ModelType='lstm'
print(ModelType)
agent = Double_DQN(state_dim=state_dim, hidden_size=hidden_size,ModelType=ModelType,MEMORY_THRESHOLD=5000)
epoch =7
for k in range(epoch):  # 开的游戏回合
    #print(k)
    env = Env(data_path="Data/15m000300/RESSET_INDXSH2018-2021_000300.csv")
    guidance=Guidance(env)
    guidanceFlag=0
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = 1
    while env.TimeCursor < env.DataLen:

        if done == 1:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
                if env.DataLen - env.TimeCursor < arg.ADayTime * 5:
                    break

            done = 0  # 游戏未开始
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            #若出现奖励不正常情况
            if math.isnan(episode_reward):
                print(episode_reward)
                print(k)
                print(env.Data.loc[env.TimeCursor,'time'])

        bar = env.Data.loc[env.TimeCursor, :]

        if done == 0:
            obs = env.Observation
            obs = obs.values
            obs = obs.astype(np.float64)
            if guidanceFlag==1:
                action=guidance.getGuidanceAction()
            else:
                action = agent.action(obs, israndom=True)


            new_obs, reward, done = env.step(action=action-1)

            new_obs = new_obs.values
            new_obs = new_obs.astype(np.float64)
            episode_reward += reward
            agent.learn(obs, action, reward, new_obs, done,env.ResistancePointFlag,env.hold_time)

            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

torch.save(agent.network.state_dict(),str(hidden_size)+'network-'+ModelType+'Noguiadance.pth')
torch.save(agent.target_network.state_dict(),str(hidden_size)+'target-network-'+ModelType+'Noguiadance.pth')
#np.save(file='LossRecord'+ModelType+'Noguiadance.npy',arr=np.array(agent.loss_record))
