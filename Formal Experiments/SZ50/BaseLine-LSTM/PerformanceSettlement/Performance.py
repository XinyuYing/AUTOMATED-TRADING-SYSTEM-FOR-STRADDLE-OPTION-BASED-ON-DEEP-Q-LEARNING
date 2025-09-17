#账户测试
import numpy as np
import pandas as pd

from Model.Critic import Critic_LSTM
from Model.Double_DQN_Load import Double_DQN
from BaseLineModel.Predict_LSTM import Predict_LSTM
from Model.Env import Env
import math
from Setting import arg
from Preprocess.SettleAccount import IndexAccount
import torch
OptionTransaction=pd.read_csv('PerformanceSettlement/HistoricalTransaction.csv')
OptionTransaction['OpenTime']=pd.to_datetime(OptionTransaction['OpenTime'])
OptionTransaction['CloseTime']=pd.to_datetime(OptionTransaction['CloseTime'])
OptionProfitrate=[]
LongProfitrate=[]
env=Env(data_path="Data/15m000300/RESSET_INDXSH2022_000300.csv")
Performance=pd.DataFrame(columns=['OptionProfitRate','LongProfitRate','ShortProfitRate','LstmProfitRate','time'])#业绩

initPoint=env.Data.loc[env.TimeCursor,'open']

indexAccount=IndexAccount()
longAccount=IndexAccount()
longAccount.open(price=initPoint,action=1)
shortAccount=IndexAccount()
shortAccount.open(price=initPoint,action=-1)

LSTMBaseLine=torch.load('ModelParm/2024-02-13LSTM.pth')
LSTMBaseLine=LSTMBaseLine.cpu()


state_dim = 10
hidden_size = 64
network=Critic_LSTM(state_dim=state_dim,hidden_size=hidden_size)
target_network=Critic_LSTM(state_dim=state_dim,hidden_size=hidden_size)
ModelType='lstm'
network.load_state_dict(torch.load('ModelParm/2024年2月10日 18-21年训练，22年测试/64network-lstmNoguiadance.pth'))
target_network.load_state_dict(torch.load('ModelParm/2024年2月10日 18-21年训练，22年测试/64target-network-lstmNoguiadance.pth'))
agent = Double_DQN(network=network,target_network=target_network)


OptionOpenFlag=0
OptionTransactionIndex=0
Data=dict()
while env.TimeCursor<env.Data.shape[0]-arg.ADayTime*2:
    print(env.TimeCursor)
    bar=env.Data.loc[env.TimeCursor,:]
    #print(env.TimeCursor)
    Assert=env.account.AllCash+env.account.getMarketValue(price=bar['close'],time=bar['time'],IV=bar['HV'])
    Data['OptionProfitRate']=(Assert/env.account.initCash)
    Data['LongProfitRate']=(longAccount.getMarketValue(bar['close'])/longAccount.initCash)
    Data['ShortProfitRate'] = (shortAccount.getMarketValue(bar['close']) / shortAccount.initCash)
    Data['LstmProfitRate']=(indexAccount.getMarketValue(bar['close'])/indexAccount.initCash)
    Data['time']=bar['time']
    new_data = pd.DataFrame(Data, index=[0])
    Performance = pd.concat([Performance, new_data], ignore_index=True, axis=0)

    obs = env.Observation
    obs = obs.values
    obs = obs.astype(np.float64)
    action = agent.action(obs, ResistancePointFlag=env.ResistancePointFlag, hold_time=env.hold_time)


    #baseline方向交易
    if env.TimeCursor%arg.ADayTime==0:
        X=env.Observation.loc[:, ['high', 'low', 'open', 'close', 'avg', 'vol', 'amount']].values
        X=torch.Tensor([X])
        Y=LSTMBaseLine.forward(X)
        if Y>0:
            if indexAccount.action==-1:
                indexAccount.close(bar['close'])
            if indexAccount.action==0:
                indexAccount.open(bar['close'],action=1)
        if Y<0:
            if indexAccount.action==1:
                indexAccount.close(bar['close'])
            if indexAccount.action==0:
                indexAccount.open(bar['close'],action=-1)



    env.step(action=action)

Performance.to_csv('Performance3_new.csv',index=False)

Performance=pd.read_csv('Performance3.csv')

import matplotlib.pyplot as plt
