import pandas as pd
import math
from Setting import arg
from statistics import mean,stdev
class MABias():#近n天的与M日均线偏离统计
    def __init__(self,Data,#股票数据
                 TimeCursor,#时间游标
                 MA_N,#MA指标周期，n日均线
                 BIAS_N#bias的统计周期
                 ):
        self.MA_N=MA_N*arg.ADayTime
        self.BIAS_N=BIAS_N*arg.ADayTime
        self.ma=Data.loc[0,'avg']
        self.bias=[0]*self.BIAS_N
        self.biasIndex=0
        self.Data=Data
        for i in range(self.BIAS_N):
            self.ma=sum(Data.loc[TimeCursor-self.MA_N+1-i:TimeCursor-i,'avg'])/self.MA_N
            self.biasIndex=(TimeCursor-i)%self.BIAS_N
            self.bias[self.biasIndex]=abs(math.log(self.Data.loc[TimeCursor-i,'avg']/self.ma))

    def getMA_bias(self,TimeCursor):

        self.ma=sum(self.Data.loc[TimeCursor-self.MA_N+1:TimeCursor,'avg'])/self.MA_N
        self.biasIndex=(TimeCursor)%self.BIAS_N
        self.bias[self.biasIndex] = abs(math.log(self.Data.loc[TimeCursor,'avg']/self.ma))
        return self.ma,self.bias


class LogHV():#
    def __init__(self, Data,  # 股票数据
                 TimeCursor,  # 时间游标
                 N,  # HV指标周期，近几日的年华波动率
                 ):
        self.Data=Data
        self.F=240*arg.ADayTime#一年的样本量
        self.N=N*arg.ADayTime#样本区间的样本量
        self.HV=[0]*self.N
        for i in range(self.N):
            self.HV[-i-1]=math.log(self.Data.loc[TimeCursor-i,'avg']/self.Data.loc[TimeCursor-i-1,'avg'])
    def getLogHV(self,TimeCursor):
        self.HV[TimeCursor%self.N]=math.log(self.Data.loc[TimeCursor,'avg']/self.Data.loc[TimeCursor-1,'avg'])
        sum=0
        for i in range(self.N):
            sum=sum+self.HV[i]**2
        return math.sqrt(self.F/self.N*sum)


class Monmentum():
    def __init__(self,Data,ShortN,#短期速度
                 LongN,#长期速度
                 EMA_N,#平滑项
                 TimeCursor):
        self.ShortN=ShortN*arg.ADayTime
        self.LongN=LongN*arg.ADayTime
        self.EMA_N=EMA_N
        self.Period=240/arg.ADayTime
        self.emaShortSpeed=(Data.loc[self.LongN,'avg']-Data.loc[self.LongN-self.ShortN,'avg'])/(self.ShortN*self.Period)
        self.emaLongSpeed =(Data.loc[self.LongN,'avg']-Data.loc[self.LongN-self.LongN,'avg']) / (self.LongN*self.Period)
        self.Data=Data
        for i in range(self.LongN,TimeCursor+1):
            shortSpeed = (Data.loc[i, 'avg'] - Data.loc[i - self.ShortN, 'avg']) / (self.ShortN*self.Period)
            longSpeed = (Data.loc[i, 'avg'] - Data.loc[i - self.LongN, 'avg']) / (self.LongN*self.Period)
            self.emaShortSpeed=(2*shortSpeed+(EMA_N-1)*self.emaShortSpeed)/(EMA_N+1)
            self.emaLongSpeed = (2 * longSpeed + (EMA_N - 1) * self.emaLongSpeed) / (EMA_N + 1)

    def getMonmentum(self,TimeCursor):
        shortSpeed = (self.Data.loc[TimeCursor, 'avg'] - self.Data.loc[TimeCursor - self.ShortN, 'avg']) / (self.ShortN*self.Period)
        longSpeed = (self.Data.loc[TimeCursor,'avg'] - self.Data.loc[TimeCursor - self.LongN, 'avg']) / (self.LongN*self.Period)
        self.emaShortSpeed = (2 * shortSpeed + (self.EMA_N - 1) * self.emaShortSpeed) / (self.EMA_N + 1)
        self.emaLongSpeed = (2 * longSpeed + (self.EMA_N - 1) * self.emaLongSpeed) / (self.EMA_N + 1)
        return self.emaShortSpeed,self.emaLongSpeed

#入场规则：靠近阻力位开仓
#出场规则：偏离开仓点位幅度达到点位2%平仓且动量绝对值死叉
if __name__ == '__main__':
    x=1
    print(x)
    Data = pd.read_csv('Data/15m000300/RESSET_INDXSH2022_000300.csv')  # 读取到的本地数据
    Data["time"] = pd.to_datetime(Data['time'])  # 送进模型的不应该包含时间和指数代码
    xx=Data.loc[640,:]
    maBias = MABias(Data=Data, TimeCursor=640, MA_N=5,BIAS_N=20)
    ma,bias=maBias.getMA_bias(TimeCursor=640)
    bias_mean = mean(bias)
    bias_std = stdev(bias)
    mon=Monmentum(Data=Data,ShortN=1,LongN=5,EMA_N=5,TimeCursor=640)
    loghv=LogHV(Data=Data,TimeCursor=640,N=5)
    yy=loghv.getLogHV(TimeCursor=640)
    Data.loc[640,'time']