import pandas as pd
from Preprocess.SettleAccount import Account
from Model.Env import Env
import math
from Setting import arg
OptionTransaction=pd.read_csv('PerformanceSettlement/HistoricalTransaction.csv')
OptionTransaction['OpenTime']=pd.to_datetime(OptionTransaction['OpenTime'])
OptionTransaction['CloseTime']=pd.to_datetime(OptionTransaction['CloseTime'])
OptionProfitrate=[]
LongProfitrate=[]
env=Env(data_path="Data/15m000300/RESSET_INDXSH2022_000300.csv")
Performance=pd.DataFrame(columns=['OptionProfitRate','LongProfitRate','time'])#业绩
initPoint=env.Data.loc[env.TimeCursor,'open']

OptionOpenFlag=0
OptionTransactionIndex=0
Data=dict()
while env.TimeCursor<env.Data.shape[0]-arg.ADayTime-1:
    bar=env.Data.loc[env.TimeCursor,:]
    #print(env.TimeCursor)
    Assert=env.account.AllCash+env.account.getMarketValue(price=bar['close'],time=bar['time'],IV=bar['HV'])
    Data['OptionProfitRate']=math.log(Assert/env.account.initCash)
    Data['LongProfitRate']=math.log(bar['close']/initPoint)
    Data['time']=bar['time']
    new_data = pd.DataFrame(Data, index=[0])
    Performance = pd.concat([Performance, new_data], ignore_index=True, axis=0)

    if OptionTransactionIndex<OptionTransaction.shape[0]:
        if OptionOpenFlag==0:
            if env.Data.loc[env.TimeCursor,'time']==OptionTransaction.loc[OptionTransactionIndex,'OpenTime']:
                OptionOpenFlag=1
        if OptionOpenFlag==1:
            if env.Data.loc[env.TimeCursor,'time']==OptionTransaction.loc[OptionTransactionIndex,'CloseTime']:
                OptionOpenFlag=0
                OptionTransactionIndex+=1

    env.step(action=OptionOpenFlag)

Performance.to_csv('Performance.csv',index=False)


import matplotlib.pyplot as plt