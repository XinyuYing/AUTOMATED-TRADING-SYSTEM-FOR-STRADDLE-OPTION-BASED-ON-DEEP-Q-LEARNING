import pandas as pd
year='2018'
file_name='RESSET_INDXSH'+year+'_000300.csv'
hs300=pd.read_csv('Data/000300/'+file_name,encoding='gbk')
date_range = pd.date_range(start=year+'-01-01', end=year+'-12-31')
hs300["time"]=pd.to_datetime(hs300["行情日期_Qdate"]+' '+hs300["标准时间_StdTime"])
hs300=hs300[['代码_Code',
                   '期间最高价(元)_Highpr',
                   '期间最低价(元)_Lowpr',
                   '期初成交价(元)_Begpr',
                   '成交价(元)_TPrice',
                   '期间累计成交量(股)_TVolume_accu1',
                   '期间累计成交额(元)_TSum_accu1',
                   'time']]

new_column={'代码_Code':'code',
            '期间最高价(元)_Highpr':'high',
            '期间最低价(元)_Lowpr':'low',
            '期初成交价(元)_Begpr':'open',
            '成交价(元)_TPrice':'avg',
            '期间累计成交量(股)_TVolume_accu1':'vol',
            '期间累计成交额(元)_TSum_accu1':'amount'}

hs300=hs300.rename(columns=new_column)
#删掉盘前和盘后的多余数据
condition11=pd.Timestamp('09:31:00').time()<=hs300['time'].dt.time
condition12=hs300['time'].dt.time<=pd.Timestamp('11:30:00').time()
condition1=condition11 & condition12

condition21=pd.Timestamp('13:01:00').time()<=hs300['time'].dt.time
condition22=hs300['time'].dt.time<=pd.Timestamp('15:00:00').time()
condition2=condition21 & condition22

condition41=pd.Timestamp('11:32:00').time()<=hs300['time'].dt.time
condition42=hs300['time'].dt.time<=pd.Timestamp('13:00:00').time()
condition4=condition41 & condition42
hs300.loc[condition4,'vol']=0

condition3=hs300['vol']==0
condition=condition3 & (condition1 | condition2)
hs300.loc[condition,'vol']=1
hs300=hs300[hs300['vol']!=0]
hs300 = hs300.reset_index(drop=True)



#合并：零散的k线进行合并



from datetime import datetime,timedelta
#洗日内数据
morning_start_time = datetime.strptime("09:30:00", "%H:%M:%S")
morning_end_time = datetime.strptime("11:30:00", "%H:%M:%S")
afternoon_start_time = datetime.strptime("13:00:00", "%H:%M:%S")
afternoon_end_time = datetime.strptime("15:00:00", "%H:%M:%S")

for date in date_range:
    date_str=date.strftime('%Y-%m-%d')
    filtered_data = hs300[hs300['time'].dt.strftime('%Y-%m-%d') == date_str]
    if filtered_data.shape[0]==0:continue
    morning_start_time = datetime.strptime(date_str+' '+"09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end_time = datetime.strptime(date_str+' '+"11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start_time = datetime.strptime(date_str+' '+"13:00:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end_time = datetime.strptime(date_str+' '+"15:00:00", "%Y-%m-%d %H:%M:%S")

    for index, row in filtered_data.iterrows():
        time = row['time']
        #不在连续竞价期间的数据直接跳过
        if not (morning_start_time<=time<=morning_end_time or afternoon_start_time<=time<=afternoon_end_time):
            continue;
        next_data=hs300.loc[index+1]
        next_time=next_data['time']

        #相隔数据不在同一天内，跳过
        if time.strftime('%Y-%m-%d')!=next_time.strftime('%Y-%m-%d'):continue

        #计算两个数据之间的时间差
        if time<=morning_end_time:
            minutes_difference=int(((min(next_time,morning_end_time+timedelta(minutes=1))-time)).total_seconds()/60)
        else:
            minutes_difference=int(((min(next_time,afternoon_end_time+timedelta(minutes=1))-time)).total_seconds()/60)

        if minutes_difference==1:continue#代表相邻数据没有异常，直接跳过
        print(next_time)

        avg_vol=int((next_data['vol']+row['vol'])/(minutes_difference+1))
        avg_amount=(next_data['amount']+row['amount'])/(minutes_difference+1)
        hs300.loc[index,'vol']=avg_vol
        hs300.loc[index,'amount']=avg_amount
        hs300.loc[index+1,'vol']=avg_vol
        hs300.loc[index+1,'amount']=avg_amount
        avg_diff_price=(next_data['avg']-row['avg'])/minutes_difference
        base_price=row['avg']
        code=row['code']


        for i in range(minutes_difference-1):
            price=base_price+(i+1)*avg_diff_price
            new_time=time+(i+1)*timedelta(minutes=1)
            new_data={'code':code,'high':price,'low':price,'open':price,"avg":price,'vol':avg_vol,'amount':avg_amount,'time':new_time}
            max_index=hs300.index.max()+1
            hs300.loc[max_index]=new_data

hs300=hs300.sort_values(by='time')
hs300=hs300.reset_index(drop=True)
grouped_data = hs300.groupby(hs300['time'].dt.date).count()
hs300.to_csv('Data/B000300/'+file_name,index=False)


