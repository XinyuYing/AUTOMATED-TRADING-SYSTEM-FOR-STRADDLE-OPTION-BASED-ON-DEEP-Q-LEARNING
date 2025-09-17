import statistics
from Preprocess.Indicator import LogHV
from Setting import arg
import pandas as pd
from Preprocess.Tool import Hurst
from Preprocess.TransferData import TransferData
from Preprocess.SettleAccount import Account
import numpy as np
import math


# 持仓标记应该打入状态
class Env():
    def __init__(self, data_path):
        self.Data = pd.read_csv(data_path)  # 读取到的本地数据
        self.Data["time"] = pd.to_datetime(self.Data['time'])  # 送进模型的不应该包含时间和指数代码

        self.Data["PositionMarker"] = 0.0  # 新添加持仓浮盈浮亏
        self.Data["NextDay"] = 1  # 下一个交易日间隔
        self.Data['HV'] = 0.16  # 波动率
        # self.Data=self.DataAll.loc[:,['high','low','open','avg','vol','amount','time']]
        self.InitDay = 25  # 不能低于Max(ModelWindow+BeforeVOLN,HV)
        self.ModelWindow = arg.history_data_len * arg.ADayTime
        self.TimeCursor = self.InitDay * arg.ADayTime  # 给模型能看到的数据时间窗口游标
        self.Time = self.Data.loc[self.TimeCursor, 'time']
        self.DataLen = self.Data.shape[0]
        self.ResistanceCursor = 0  # 数据计算阻力位的窗口游标
        self.headTtoTailDifference1 = 0  # 首尾差1
        self.headTtoTailDifference2 = 0  # 首尾差1
        self.down_flag = 0  # 下降趋势标记
        self.up_flag = 0  # 上升趋势标记
        self.before_action = 0  # 前一个时刻动作
        self.account = Account()  # 账户
        self.hold_time = 0  # 持有时间
        self.open_point = 0
        self.ResistancePointFlag = 0  # 标记价格是否运动到了阻力位

        self.ResistancePoint = [[self.ResistanceCursor, self.Data.loc[self.ResistanceCursor, 'time'],
                                 self.Data.loc[self.ResistanceCursor, 'avg']]]  # 阻力位
        self.SupportPoint = [[self.ResistanceCursor, self.Data.loc[self.ResistanceCursor, 'time'],
                              self.Data.loc[self.ResistanceCursor, 'avg']]]  # 支撑位

        # 波动率
        self.loghv = LogHV(Data=self.Data, TimeCursor=5 * arg.ADayTime + 1, N=5)
        for i in range(5 * arg.ADayTime + 1, self.TimeCursor + 1):
            self.HV = self.loghv.getLogHV(i)
            self.Data.loc[i, 'HV'] = self.HV

        self.transfer = TransferData(Data=self.Data, TimeCursor=self.TimeCursor, BeforeN=self.ModelWindow, BeforeVOLN=5)
        self.Observation = self.transfer.DataBuffer[
            ['high', 'low', 'open', 'close', 'avg', 'vol', 'amount', 'PositionMarker', 'NextDay', 'HV']]
        self.getResistanceSupport()

        self.HistoricalTransaction = pd.DataFrame(columns=['OpenTime', 'OpenPoint', 'OpenHV',
                                                           'CloseTime', 'ClosePoint', 'CloseHV', 'ProfitRate'])
        self.Order = dict()

    def getResistanceSupport(self):  # 获得支撑阻力
        windows = arg.window
        while self.ResistanceCursor < self.TimeCursor - windows + 1:
            fragment = self.Data.loc[self.ResistanceCursor:self.ResistanceCursor + windows, 'avg'].values
            self.headTtoTailDifference2 = fragment[-1] - fragment[0]
            # print(self.ResistanceCursor)

            if self.headTtoTailDifference2 * self.headTtoTailDifference1 < 0:
                ReturnFragment = self.Data.loc[self.ResistanceCursor - 1:self.ResistanceCursor + windows, :]

                if self.headTtoTailDifference2 > 0:  # 下跌见底
                    Index = ReturnFragment.idxmin(axis=0).loc['avg']
                    ReturnPoint = self.Data.loc[Index, "avg"]
                    time = self.Data.loc[Index, "time"]

                    if Index != self.SupportPoint[-1][0]:  # 找到的阻力位不能和之前是重复的
                        if self.down_flag == 1:  # 持续下跌
                            basePoint = self.SupportPoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) < -0.01:
                                self.SupportPoint.append([Index, time, ReturnPoint])
                        else:  # 从顶部下跌
                            basePoint = self.ResistancePoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) < -0.015:
                                self.SupportPoint.append([Index, time, ReturnPoint])
                                self.down_flag = 1
                                self.up_flag = 0

                if self.headTtoTailDifference2 < 0:  # 上涨见顶
                    Index = ReturnFragment.idxmax(axis=0).loc['avg']
                    ReturnPoint = self.Data.loc[Index, "avg"]
                    time = self.Data.loc[Index, "time"]

                    if Index != self.ResistancePoint[-1][0]:  # 找到的阻力位不能和之前是重复的
                        if self.up_flag == 1:  # 持续上涨
                            basePoint = self.ResistancePoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) > 0.01:
                                self.ResistancePoint.append([Index, time, ReturnPoint])
                        else:  # 从底部涨，
                            basePoint = self.SupportPoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) > 0.015:
                                self.ResistancePoint.append([Index, time, ReturnPoint])
                                self.down_flag = 0
                                self.up_flag = 1
            self.ResistanceCursor = self.ResistanceCursor + 1
            self.headTtoTailDifference1 = self.headTtoTailDifference2

    def Outliers_detection(self):  # 异常值检测
        win = 3  # 前后振幅
        ResistancePointFrontAmplitudes = []
        ResistancePointAfterAmplitudes = []
        # 压力位的振幅过滤
        for i in range(1, len(self.ResistancePoint)):
            MaxPoint = max(self.Data.loc[int(self.ResistancePoint[i][0]) - win:int(self.ResistancePoint[i][0]), "avg"])
            MinPoint = min(self.Data.loc[int(self.ResistancePoint[i][0]) - win:int(self.ResistancePoint[i][0]), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            ResistancePointFrontAmplitudes.append(FrontAmplitude)

            MaxPoint = max(self.Data.loc[int(self.ResistancePoint[i][0]):int(self.ResistancePoint[i][0] + win), "avg"])
            MinPoint = min(self.Data.loc[int(self.ResistancePoint[i][0]):int(self.ResistancePoint[i][0] + win), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            ResistancePointAfterAmplitudes.append(FrontAmplitude)

        i = 1
        j = 0
        StdFrontAmplitudes = statistics.stdev(ResistancePointFrontAmplitudes)
        MeanFrontAmplitudes = statistics.mean(ResistancePointFrontAmplitudes)
        StdAfterAmplitudes = statistics.stdev(ResistancePointAfterAmplitudes)
        MeanAfterAmplitudes = statistics.mean(ResistancePointAfterAmplitudes)
        while i < len(self.ResistancePoint):
            if ResistancePointFrontAmplitudes[j] > MeanFrontAmplitudes + 3 * StdFrontAmplitudes:
                if ResistancePointAfterAmplitudes[j] > MeanAfterAmplitudes + 3 * StdAfterAmplitudes:
                    del self.ResistancePoint[i]
                    i = i - 1
            i = i + 1
            j = j + 1
        SupportPointFrontAmplitudes = []
        SupportPointAfterAmplitudes = []
        # 阻力位的振幅过滤
        for i in range(1, len(self.SupportPoint)):
            MaxPoint = max(self.Data.loc[int(self.SupportPoint[i][0]) - win:int(self.SupportPoint[i][0]), "avg"])
            MinPoint = min(self.Data.loc[int(self.SupportPoint[i][0]) - win:int(self.SupportPoint[i][0]), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            SupportPointFrontAmplitudes.append(FrontAmplitude)

            MaxPoint = max(self.Data.loc[int(self.SupportPoint[i][0]):int(self.SupportPoint[i][0] + win), "avg"])
            MinPoint = min(self.Data.loc[int(self.SupportPoint[i][0]):int(self.SupportPoint[i][0] + win), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            SupportPointAfterAmplitudes.append(FrontAmplitude)

        i = 1
        j = 0
        StdFrontAmplitudes = statistics.stdev(SupportPointFrontAmplitudes)
        MeanFrontAmplitudes = statistics.mean(SupportPointFrontAmplitudes)
        StdAfterAmplitudes = statistics.stdev(SupportPointAfterAmplitudes)
        MeanAfterAmplitudes = statistics.mean(SupportPointAfterAmplitudes)
        while i < len(self.SupportPoint):
            if SupportPointFrontAmplitudes[j] > MeanFrontAmplitudes + 3 * StdFrontAmplitudes:
                if SupportPointAfterAmplitudes[j] > MeanAfterAmplitudes + 3 * StdAfterAmplitudes:
                    del self.SupportPoint[i]

                    i = i - 1
            i = i + 1
            j = j + 1
        return

    def getResistancePointFlag(self, beforeN):  # 价格运动到阻力位区间则发出1的信号,否则为0
        bar = self.Data.loc[self.TimeCursor, :]
        for j in range(beforeN):
            if j > len(self.ResistancePoint) or j > len(self.SupportPoint): break
            if abs(math.log(bar['avg'] / self.ResistancePoint[-j][-1])) < 0.003:
                self.ResistancePointFlag = 1
                return
            if abs(math.log(bar['avg'] / self.SupportPoint[-j][-1])) < 0.003:
                self.ResistancePointFlag = 1
                return
        self.ResistancePointFlag = 0
        return

    def step(self, action=0.6):  # 播放下一根k线，给模型看下一个时间状态,对应的奖励，以及该轮游戏是否结束
        action = round(action)
        bar = self.Data.loc[self.TimeCursor, :]
        beforeHV = self.HV
        self.TimeCursor = self.TimeCursor + 1
        next_bar = self.Data.loc[self.TimeCursor, :]
        self.HV = self.loghv.getLogHV(self.TimeCursor)
        self.Data.loc[self.TimeCursor, 'HV'] = self.HV
        if self.before_action == 1:  # 如果上一时刻为持有，更新浮盈浮亏
            self.Data.loc[self.TimeCursor, 'PositionMarker'] = math.log(self.account.getMarketValue(price=next_bar['close'], time=next_bar['time'],IV=self.HV) / self.account.OpenMarketValue)

        done = -1  # 游戏没开始
        reward = 0
        if action == 1 and self.before_action == 0:  # 开仓
            self.account.OpenPosition(price=bar['close'], time=bar['time'], IV=beforeHV)

            self.Order['OpenTime'] = bar['time']
            self.Order['OpenPoint'] = bar['close']
            self.Order['OpenHV']=beforeHV
            self.Data.loc[self.TimeCursor, 'PositionMarker'] = math.log(self.account.getMarketValue(price=next_bar['close'], time=next_bar['time'],IV=self.HV) / self.account.OpenMarketValue)
            self.hold_time = 1
            self.open_point = bar['close']
            reward = 0
            done = 0
        if action == 0 and self.before_action == 1:  # 平仓
            self.account.ClosePosition(price=bar['close'], time=bar['time'], IV=beforeHV)

            self.Order['CloseTime'] = bar['time']
            self.Order['ClosePoint'] = bar['close']
            self.Order['CloseHV'] = beforeHV
            self.Order['ProfitRate']=bar['PositionMarker']
            transaction=pd.DataFrame(self.Order,index=[0])
            self.HistoricalTransaction = pd.concat([self.HistoricalTransaction, transaction], ignore_index=True, axis=0)
            self.Order=dict()

            reward = self.reward_fun(action)
            self.Data.loc[:, 'PositionMarker'] = 0.0
            self.transfer.DataBuffer.loc[:, 'PositionMarker'] = 0.0  # 平仓后浮盈浮亏清0
            done = 1
        if action == 1 and self.before_action == 1:  # 持仓
            reward = self.reward_fun(action)
            done = 0
        if action == 0 and self.before_action == 0:  # 空仓,配合无规则使用
            reward = 0
            done = 0

        self.before_action = action
        self.getResistancePointFlag(beforeN=5)

        TempObservation = self.transfer.OrdinaryToLog(
            self.TimeCursor)  # 由于是间接更新，当一个交易周期结束后并不影响里面的缓存，应该全部吧buffer里面的Positionmarket清0
        self.Observation = TempObservation[
            ['high', 'low', 'open', 'close', 'avg', 'vol', 'amount', 'PositionMarker', 'NextDay', 'HV']]

        if self.TimeCursor % arg.ADayTime == 0:  # 过完了一天，更新阻力位
            self.getResistanceSupport()

        return self.Observation, reward, done

    def reward_fun(self, action):
        # 激励函数应该与交易动作，开仓价格与阻力位之间的距离，持仓时间有关
        # 奖励等完成开仓，持仓，平仓一个周期后才进行结算，开仓和持仓都不给任何奖励
        # 错失机会给小惩罚
        reward = 0

        if action == 1 and self.before_action == 1:  # 持有过程，如果触及止损线，立刻平仓
            bar = self.Data.loc[self.TimeCursor, :]
            self.hold_time = self.hold_time + 1
            if bar['PositionMarker'] < -0.15:
                reward = math.exp(bar['PositionMarker']) - 1  # 提示浮亏

        if action == 0 and self.before_action == 1:  # 平仓
            bar = self.Data.loc[self.TimeCursor - 1, :]
            if bar['PositionMarker'] < -0.15:  # 打止损平仓，动作正确，给予奖励
                reward = 0.01
            else:
                reward = (math.exp(bar['PositionMarker']) - 1)  # 正常平仓，结算盈亏
            # if self.hold_time<0.5*arg.ADayTime:#持仓过短，直接给一个小惩罚
            #     reward=-0.1
            if abs(math.log(bar['close'] / self.open_point)) > 0.015:  # 波动超过标准，奖励加倍
                reward = 2 * reward

            self.hold_time = 0

        return reward


if __name__ == '__main__':

    env = Env(data_path="Data/15m000300/RESSET_INDXSH2022_000300.csv")
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)  # 这一行执行有问题，Hv下降导致亏损
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=0)

    for i in range(100):
        xxx = env.step()
    buffer = []

