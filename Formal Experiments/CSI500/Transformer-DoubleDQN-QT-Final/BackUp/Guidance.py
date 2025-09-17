from Model.Env import Env
from Preprocess.SettleAccount import Account
from Preprocess.Indicator import Monmentum,MABias
import math
from Setting import arg
from statistics import mean,stdev


class Guidance():
    def __init__(self,env:Env):
        self.env=env
        self.account=env.account
        #技术指标
        self.maBias=MABias(Data=env.Data,TimeCursor=env.TimeCursor,MA_N=5,BIAS_N=20)
        self.monmentum=Monmentum(Data=env.Data,ShortN=1,LongN=5,EMA_N=5,TimeCursor=env.TimeCursor)
        #参数
        self.OpenFlag=0
        self.before_n=3
        self.bias_mean=0
        self.bias_std=0
        self.OpenPrice=0
        self.maxBias=0#记录达到过的最大偏离度

        self.result = []
        self.order = []


    def getGuidanceAction(self):
        bar=self.env.Data.loc[self.env.TimeCursor,:]
        nextbar = self.env.Data.loc[self.env.TimeCursor+1, :]

        #更新指标
        shortMon,longMon=self.monmentum.getMonmentum(TimeCursor=self.env.TimeCursor)
        ma,bias=self.maBias.getMA_bias(TimeCursor=self.env.TimeCursor)

        #开仓规则
        if self.OpenFlag == 0 and self.env.HV<0.30:
            for j in range(self.before_n):
                # 接近前n个阻力位任意一个就开仓
                if abs(math.log(bar['avg'] / self.env.ResistancePoint[-j][-1])) < 0.003:
                    self.order.append(bar['time'])
                    self.bias_mean = mean(bias)
                    self.bias_std = stdev(bias)
                    self.OpenFlag = 1
                    self.OpenPrice = bar['avg']
                    return self.OpenFlag
                if abs(math.log(bar['avg'] / self.env.SupportPoint[-j][-1])) < 0.003:
                    self.order.append(bar['time'])
                    self.bias_mean = mean(bias)
                    self.bias_std = stdev(bias)
                    self.OpenFlag = 1
                    self.OpenPrice = bar['avg']
                    return self.OpenFlag


        #平仓规则
        if self.OpenFlag == 1:
            marketValue = self.account.getMarketValue(price=bar['close'],time=bar['time'],IV=self.env.HV)
            openMarketValue=self.account.OpenMarketValue
                # 对数收益率偏离最大值达到两个sigma以上且短期动量小于长期动量
            if abs(math.log(bar['avg'] / self.OpenPrice))>self.maxBias:
                self.maxBias=abs(math.log(bar['avg'] / self.OpenPrice))

            if self.maxBias > abs(self.bias_mean) + 2 * self.bias_std:
                if (shortMon>0 and shortMon < longMon) or (shortMon<0 and shortMon > longMon):
                    self.order.append(bar['time'])
                    self.order.append(math.log(marketValue/openMarketValue))
                    self.result.append(self.order)
                    self.order=[]
                    self.OpenFlag = 0
                    self.maxBias=0
                    return self.OpenFlag
            elif math.log(marketValue/openMarketValue)<-0.12 or (nextbar['time']-bar['time']).days>4:
                self.order.append(bar['time'])
                self.order.append(math.log(marketValue / openMarketValue))
                self.result.append(self.order)
                self.order = []
                self.OpenFlag = 0
                self.maxBias=0
                return self.OpenFlag
            elif self.env.hold_time>116:
                self.order.append(bar['time'])
                self.order.append(math.log(marketValue / openMarketValue))
                self.result.append(self.order)
                self.order = []
                self.OpenFlag = 0
                self.maxBias=0
                return self.OpenFlag

        return self.OpenFlag




import pandas as pd
if __name__ == '__main__':
    env = Env(data_path="Data/15m000905/RESSET_INDXSH2022_000905.csv")
    guidance=Guidance(env)
    performance = pd.DataFrame(columns=['ProfitRate', 'Time'])
    while env.TimeCursor<env.DataLen-arg.ADayTime*2:
        new_data=dict()
        bar=env.Data.loc[env.TimeCursor,:]
        Assert=env.account.AllCash+env.account.getMarketValue(price=bar['close'],time=bar['time'],IV=env.HV)
        new_data['ProfitRate'] = Assert / env.account.initCash
        new_data['Time'] = bar['time']
        new_Data=pd.DataFrame(new_data,index=[0])
        performance=pd.concat([performance,new_Data],ignore_index=True,axis=0)
        action=guidance.getGuidanceAction()
        #print(action)

        env.step(action)

    performance.to_csv('Performance.csv',index=False)

    r1=[x[-1] for x in guidance.result]
    print((sum(r1)))










