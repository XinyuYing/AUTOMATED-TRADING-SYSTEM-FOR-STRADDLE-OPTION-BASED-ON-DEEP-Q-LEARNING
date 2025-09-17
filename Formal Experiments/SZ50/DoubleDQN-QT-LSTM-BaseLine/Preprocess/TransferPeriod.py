#1m K线转换为其他周期
import pandas as pd
import numpy as np
year='2018'
period=15#转换周期：5m,15m,20m,30m,60m,
file_name='RESSET_INDXSH'+year+'_000300.csv'
Data=pd.read_csv('Data/B000300/'+file_name)
date_range = pd.date_range(start=year+'-01-01', end=year+'-12-31')
Data["time"]=pd.to_datetime(Data["time"])
col=list(Data.columns.values)
col.insert(4,'close')
TransferedData=pd.DataFrame(columns=col)
TransferedData['close']
LenK=int(240/period)
for date in date_range:
    date_str=date.strftime('%Y-%m-%d')
    filtered_data = Data[Data['time'].dt.strftime('%Y-%m-%d') == date_str].copy()
    filtered_data.reset_index(drop=True,inplace=True)
    if filtered_data.shape[0] == 0: continue
    for i in range(LenK):
        index=(i+1)*period
        code=filtered_data.loc[index-period+1,'code']
        new_open=filtered_data.loc[index-period+1,'open']
        new_close=filtered_data.loc[index+1,'open']
        new_high=max(filtered_data.loc[index-period+1:index,'high'])
        new_low = min(filtered_data.loc[index - period + 1:index, 'low'])
        new_avg=sum(filtered_data.loc[index-period+1:index,'avg'])/period
        new_vol=sum(filtered_data.loc[index - period + 1:index, 'vol'])
        new_amount = sum(filtered_data.loc[index - period + 1:index, 'amount'])
        new_time=filtered_data.loc[index,'time']
        if i==0:
            new_open=filtered_data.loc[index-period,'avg']
            new_vol+=filtered_data.loc[index-period,'vol']
            new_amount += filtered_data.loc[index - period, 'amount']
        if i==LenK-1:
            new_close=filtered_data.loc[index+1,'avg']
            new_vol+=filtered_data.loc[index+1,'vol']
            new_amount += filtered_data.loc[index+1, 'amount']
        new_data = {'code': code, 'high': new_high, 'low': new_low, 'open': new_open,'close':new_close,'vol': new_vol,'avg':new_avg,
                    'amount': new_amount, 'time': new_time}
        new_data = pd.DataFrame(new_data, index=[0])
        TransferedData = pd.concat([TransferedData, new_data], ignore_index=True, axis=0)

TransferedData.to_csv('./Data/15m000300/'+file_name,index=False)





