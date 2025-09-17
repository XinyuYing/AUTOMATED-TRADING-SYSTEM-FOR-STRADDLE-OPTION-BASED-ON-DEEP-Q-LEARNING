#评价指标：夏普比率
import pandas as pd
import math
Data=pd.read_csv('Performance2.csv')
Data=pd.read_csv('Performance3降低开仓比例.csv')
Data['OptionProfitRate']=Data['OptionProfitRate'].apply(lambda x:math.log(x))
Data['LstmProfitRate']=Data['LstmProfitRate'].apply(lambda x:math.log(x))
Data['LongProfitRate']=Data['LongProfitRate'].apply(lambda x:math.log(x))
Data['ShortProfitRate']=Data['ShortProfitRate'].apply(lambda x:math.log(x))
Data1=pd.DataFrame(columns=Data.columns)#每日收益率
for i in range(1,Data.shape[0]):
    new_data=dict()
    new_data['OptionProfitRate']=Data.loc[i,'OptionProfitRate']-Data.loc[i-1,'OptionProfitRate']
    new_data['LstmProfitRate'] = Data.loc[i, 'LstmProfitRate'] - Data.loc[i - 1, 'LstmProfitRate']
    new_data['LongProfitRate'] = Data.loc[i, 'LongProfitRate'] - Data.loc[i - 1, 'LongProfitRate']
    new_data['ShortProfitRate'] = Data.loc[i, 'ShortProfitRate'] - Data.loc[i - 1, 'ShortProfitRate']
    new_data['time'] = Data.loc[i, 'time']
    new_data = pd.DataFrame(new_data, index=[0])
    Data1 = pd.concat([Data1, new_data], ignore_index=True, axis=0)

#计算夏普比
from statistics import stdev

OptionSP=(Data.loc[Data.shape[0]-1,'OptionProfitRate'])/(stdev(Data1.loc[:,'OptionProfitRate'])*math.sqrt(3840))
LSTMSP=(Data.loc[Data.shape[0]-1,'LstmProfitRate'])/(stdev(Data1.loc[:,'LstmProfitRate'])*math.sqrt(3840))
LongSP=(Data.loc[Data.shape[0]-1,'LongProfitRate'])/(stdev(Data1.loc[:,'LongProfitRate'])*math.sqrt(3840))
ShortSP=(Data.loc[Data.shape[0]-1,'ShortProfitRate'])/(stdev(Data1.loc[:,'ShortProfitRate'])*math.sqrt(3840))


#最大回撤
MaxRecallOption=0
MaxRecallLstm=0
MaxRecallLong=0
MaxRecallShort=0

RecallOption=0
RecallLstm=0
RecallLong=0
RecallShort=0
for i in range(1,Data.shape[0]-1):
    RecallOption=min(0,RecallOption+Data1.loc[i,'OptionProfitRate'])
    RecallLstm = min(0, RecallOption + Data1.loc[i, 'LstmProfitRate'])
    RecallLong = min(0, RecallOption + Data1.loc[i, 'LongProfitRate'])
    RecallShort = min(0, RecallOption + Data1.loc[i, 'ShortProfitRate'])

    if RecallOption<MaxRecallOption:
        MaxRecallOption=RecallOption

    if RecallLstm<MaxRecallLstm:
        MaxRecallLstm=RecallLstm

    if RecallLong<MaxRecallLong:
        MaxRecallLong=RecallLong

    if RecallShort<MaxRecallShort:
        MaxRecallShort=RecallShort
