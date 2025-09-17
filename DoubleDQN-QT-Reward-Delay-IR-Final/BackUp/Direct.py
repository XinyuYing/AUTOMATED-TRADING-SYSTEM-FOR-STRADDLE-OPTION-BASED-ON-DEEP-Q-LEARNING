from Model.Env import Env
from Preprocess.SettleAccount import Account
from Preprocess.Indicator import Monmentum,LogHV,MABias
import math
from Setting import arg
from statistics import mean,stdev
env = Env(data_path="Data/15m000300/RESSET_INDXSH2021_000300.csv")
account=Account()

bar=env.Data.loc[env.TimeCursor,:]
#参数指标
ma_bias=MABias(Data=env.Data,TimeCursor=env.TimeCursor,MA_N=5,BIAS_N=20)
ma,bias=ma_bias.getMA_bias(TimeCursor=env.TimeCursor)

monmentum=Monmentum(Data=env.Data,ShortN=1,LongN=5,EMA_N=5,TimeCursor=env.TimeCursor)
shortMon,longMon=monmentum.getMonmentum(TimeCursor=env.TimeCursor)

loghv=LogHV(Data=env.Data,TimeCursor=env.TimeCursor,N=5)
HV=env.HV
maxBias=0




before_n=3
OpenFlag=0
OpenAssert=0
OpenPrice=0
bias_mean=mean(bias)
bias_std=stdev(bias)
result=[]
order=[]
while env.TimeCursor<env.DataLen-arg.ADayTime-1:
    bar=env.Data.loc[env.TimeCursor,:]
    nextbar = env.Data.loc[env.TimeCursor+1, :]
    #更新指标
    shortMon,longMon=monmentum.getMonmentum(TimeCursor=env.TimeCursor)
    ma,bias=ma_bias.getMA_bias(TimeCursor=env.TimeCursor)
    HV = env.HV

    if OpenFlag==0:
        for j in range(before_n):
            # 接近前n个阻力位任意一个就开仓
            if abs(math.log(bar['avg'] / env.ResistancePoint[-j][-1])) < 0.003:
                order.append(bar['time'])
                print(bar['time'])
                account.OpenPosition(price=bar['close'], time=bar['time'],IV=HV)
                bias_mean = mean(bias)
                bias_std = stdev(bias)
                OpenFlag = 1
                OpenPrice=bar['avg']
                OpenAssert = account.getMarketValue(price=bar['close'], time=bar['time'],IV=HV)
                break
            if abs(math.log(bar['avg'] / env.SupportPoint[-j][-1])) < 0.003:
                order.append(bar['time'])
                print(bar['time'])
                account.OpenPosition(price=bar['close'], time=bar['time'],IV=HV)
                bias_mean = mean(bias)
                bias_std = stdev(bias)
                OpenFlag = 1
                OpenPrice = bar['avg']
                OpenAssert = account.getMarketValue(price=bar['close'], time=bar['time'],IV=HV)
                break
    if OpenFlag == 1:
        Assert = account.getMarketValue(price=bar['close'], time=bar['time'],IV=HV)
        # 对数收益率偏离达到两个sigma以上且短期动量小于长期动量(下跌时反过来)
        if abs(math.log(bar['avg'] / OpenPrice)) > maxBias:
            maxBias = abs(math.log(bar['avg'] / OpenPrice))

        if maxBias > abs(bias_mean) + 2 * bias_std:
            if (shortMon>0 and shortMon < longMon) or (shortMon<0 and shortMon > longMon):
                order.append(bar['time'])
                OpenFlag = 0
                order.append(math.log(Assert / OpenAssert))
                result.append(order)
                account.ClosePosition(price=bar['close'], time=bar['time'], IV=HV)
                order=[]
                maxBias=0
                print(bar['time'])

        elif math.log(Assert/OpenAssert)<-0.12 or (nextbar['time']-bar['time']).days>4:
            order.append(bar['time'])
            OpenFlag = 0
            order.append(math.log(Assert / OpenAssert))
            result.append(order)
            account.ClosePosition(price=bar['close'], time=bar['time'], IV=HV)
            order = []
            maxBias=0
            print(bar['time'])

    env.step(action=0.5)






r1=[x[-1] for x in result]
print((sum(r1)))







