import math
import numpy as np
from Model.Env import Env
from Model.Double_DQN import Double_DQN
from Setting import arg
import torch
state_dim = 9
hidden_size = state_dim * 2

agent = Double_DQN(state_dim=state_dim, hidden_size=hidden_size)
epoch = 5
for k in range(epoch):  # 开的游戏回合
    print(k)
    env = Env(data_path="Data/B000300/RESSET_INDXSH2022_000300.csv")
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = 1
    before_n = 3
    while env.TimeCursor < env.DataLen:
        if done == 1:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
                if env.DataLen - env.TimeCursor < arg.ADayTime * 20:
                    break

            done = -1  # 游戏未开始
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        bar = env.Data.loc[env.TimeCursor, :]
        if done == -1:
            flag = 0
            for j in range(before_n):
                # 接近前n个阻力位任意一个就开仓
                if abs(math.log(bar['avg'] / env.ResistancePoint[-j][-1])) < 0.001:
                    obs, reward, done = env.step(action=1)
                    done = 0
                    flag = 1
                    break
                if abs(math.log(bar['avg'] / env.SupportPoint[-j][-1])) < 0.001:
                    obs, reward, done = env.step(action=1)
                    done = 0
                    flag = 1
                    break
            if flag == 0:
                obs, reward, done = env.step(action=0)
            continue

        if done == 0:
            obs = env.Observation
            if env.hold_time > 12:  # 游戏开始，开始持仓,持仓大于180分钟
                obs = obs.values
                obs = obs.astype(np.float64)
                action = agent.action(obs, israndom=True)
                # print(action)
                new_obs, reward, done = env.step(action)
                new_obs = new_obs.values
                new_obs = new_obs.astype(np.float64)
                episode_reward += reward
                agent.learn(obs, action, reward, new_obs, done)
                obs = new_obs
                episode_timesteps += 1
                total_timesteps += 1
                timesteps_since_eval += 1

            else:
                action = 1
                new_obs, reward, done = env.step(action)

torch.save(agent.network,'network-lstm.pth')
torch.save(agent.target_network,'target-network-lstm.pth')