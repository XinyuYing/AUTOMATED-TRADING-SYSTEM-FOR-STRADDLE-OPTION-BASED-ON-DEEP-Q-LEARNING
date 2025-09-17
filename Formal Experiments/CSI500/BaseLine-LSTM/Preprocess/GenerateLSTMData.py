#生产BaseLine LSTM所需要的数据样本
import pandas as pd
from Setting import arg
import torch
from Model.Env import Env
import math
env=Env('Data/15m000300/RESSET_INDXSH2022_000300.csv')
WindowTime=arg.history_data_len * arg.ADayTime
X=torch.FloatTensor(env.Observation.loc[:,['high','low','open','close','avg','vol','amount']].values)
Y=torch.FloatTensor([math.log(env.Data.loc[env.TimeCursor+arg.ADayTime,'close']/env.Data.loc[env.TimeCursor,'close'])])
X=torch.unsqueeze(X,dim=0)
Y=torch.unsqueeze(Y,dim=0)
while env.TimeCursor<env.Data.shape[0]-2*arg.ADayTime:
    for i in range(arg.ADayTime):
        env.step(action=0)

    X1=torch.FloatTensor(env.Observation.loc[:,['high','low','open','close','avg','vol','amount']].values)
    X1 = torch.unsqueeze(X1, dim=0)
    X=torch.concat([X,X1],dim=0)

    Y1 = torch.FloatTensor([math.log(env.Data.loc[env.TimeCursor + arg.ADayTime, 'close'] / env.Data.loc[env.TimeCursor, 'close'])])
    Y1 = torch.unsqueeze(Y1, dim=0)
    Y = torch.concat([Y, Y1], dim=0)

torch.save([X,Y],'Data/BaseLineLSTMData/test.pt')


X,Y=torch.load('Data/BaseLineLSTMData/test.pt')


# from torch.utils.data import Dataset,DataLoader
# class LSTMDataset(Dataset):
#     def __init__(self, X,Y):
#         self.X=X
#         self.Y=Y
#     def __getitem__(self, index):
#         # 根据索引获取样本
#         return self.X[index,:,:],self.Y[index]
#
#     def __len__(self):
#         # 返回数据集大小
#         return len(self.X)
#
#
# Data=LSTMDataset(X=X,Y=Y)
# train_loader = DataLoader(Data, batch_size=64)
# for X,Y in train_loader:
#     X1=X
#     Y1=Y

