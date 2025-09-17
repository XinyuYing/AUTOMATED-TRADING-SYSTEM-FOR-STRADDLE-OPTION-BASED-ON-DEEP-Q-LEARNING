import math
import numpy as np
import torch

from Model.Env import Env
from Model.Double_DQN import Double_DQN
from Setting import arg
from BackUp.Guidance import Guidance

state_dim = 10
hidden_size = 64
ModelType='lstm'
print(ModelType)
agent = Double_DQN(state_dim=state_dim, hidden_size=hidden_size,ModelType=ModelType)
epoch = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for k in range(epoch):  # 开的游戏回合
    #print(k)
    env = Env(data_path="Data/15m000300/RESSET_INDXSH2022_000300.csv")
    guidance=Guidance(env)
    guidanceFlag=1
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = 1
    while env.TimeCursor < env.DataLen:
        if done == 1:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f,Time:%s") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward,env.Data.loc[env.TimeCursor,'time']))
                if env.DataLen - env.TimeCursor < arg.ADayTime * 5:
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
            if guidanceFlag==1:
                action=guidance.getGuidanceAction()
            else:
                action = agent.action(obs, israndom=True,ResistancePointFlag=env.ResistancePointFlag,hold_time=env.hold_time)
            #print(action)
            new_obs, reward, done, = env.step(action)
            new_obs = new_obs.values
            new_obs = new_obs.astype(np.float64)
            episode_reward += reward
            agent.learn(obs, action, reward, new_obs, done,env.ResistancePointFlag,env.hold_time)
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

torch.save(agent.network.state_dict(),str(hidden_size)+'network-'+ModelType+'.pth')
torch.save(agent.target_network.state_dict(),str(hidden_size)+'target-network-'+ModelType+'.pth')
np.save(file='LossRecord'+ModelType+'.npy',arr=np.array(agent.loss_record))

