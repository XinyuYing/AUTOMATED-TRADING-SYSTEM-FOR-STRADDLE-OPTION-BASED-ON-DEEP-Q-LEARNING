import pandas as pd
from datetime import datetime,timedelta

year='2018-2021'
file_name='RESSET_INDXSH'+year+'_000300.csv'
hs300 = pd.read_csv('Data/15m000300/' + file_name)
date_range = pd.date_range(start=year+'-01-01', end=year+'-12-31')
hs300["time"]=pd.to_datetime(hs300['time'])
grouped_data = hs300.groupby(hs300['time'].dt.date).count()
# 洗数据数据突然就没了,后边继续补,分为缺头部分和缺尾部分2019-4-29上午缺头，2018-8-29下午缺头，2020-2-3 下午缺尾，2019-5-28上午缺头

#去掉所有11：31的数据
for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    filtered_data = hs300[hs300['time'].dt.strftime('%Y-%m-%d') == date_str]
    if filtered_data.shape[0] == 0: continue
    morning_start_time = datetime.strptime(date_str + ' ' + "09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end_time = datetime.strptime(date_str + ' ' + "11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start_time = datetime.strptime(date_str + ' ' + "13:00:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end_time = datetime.strptime(date_str + ' ' + "15:00:00", "%Y-%m-%d %H:%M:%S")

    for index, row in filtered_data.iterrows():
        time = row['time']
        if time==morning_end_time+timedelta(minutes=1):
            hs300.drop(index=index,inplace=True)
            hs300=hs300.reset_index(drop=True)
            print(time)






#补下午缺尾巴
for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    filtered_data = hs300[hs300['time'].dt.strftime('%Y-%m-%d') == date_str]
    if filtered_data.shape[0] == 0: continue
    morning_start_time = datetime.strptime(date_str + ' ' + "09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end_time = datetime.strptime(date_str + ' ' + "11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start_time = datetime.strptime(date_str + ' ' + "13:00:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end_time = datetime.strptime(date_str + ' ' + "15:00:00", "%Y-%m-%d %H:%M:%S")

    for index, row in filtered_data.iterrows():
        time = row['time']
        code = row['code']
        if index == hs300.index.max(): continue
        next_data = hs300.loc[index + 1]
        next_time = next_data['time']

        # 相隔数据不在同一天内，(应该改成不在同一个连续竞价内)
        if time.strftime('%Y-%m-%d') == next_time.strftime('%Y-%m-%d') or time > afternoon_end_time:continue
        print("****"+str(next_time))

        # 计算两个数据之间的时间差
        if time <= morning_end_time:  # 从早上开始缺
            index1 = index
            minutes_difference = int((morning_end_time - time).total_seconds() / 60) + 121
            avg_diff_price = (next_data['open'] - row['avg']) / minutes_difference
            new_price = row['avg']
            moring_lack = int((morning_end_time  - time).total_seconds() / 60)
            for i in range(moring_lack):
                new_price = new_price + avg_diff_price
                new_vol = (hs300.loc[index1 - 242, 'vol'] + hs300.loc[index1 - 242 * 2, 'vol'] + hs300.loc[
                    index1 - 242 * 3, 'vol']) / 3  # 前三天量的均值
                new_amount = (hs300.loc[index1 - 242, 'amount'] + hs300.loc[index1 - 242 * 2, 'amount'] + hs300.loc[
                    index1 - 242 * 3, 'amount']) / 3  # 前三天量的均值
                index1 = index1 + 1
                new_time = time + (i + 1) * timedelta(minutes=1)
                new_data = {'code': code, 'high': new_price, 'low': new_price, 'open': new_price, "avg": new_price,
                            'vol': new_vol, 'amount': new_amount, 'time': new_time}
                max_index = hs300.index.max() + 1
                hs300.loc[max_index] = new_data
            for i in range(121):
                new_price = new_price + avg_diff_price
                new_vol = (hs300.loc[index1 - 242, 'vol'] + hs300.loc[index1 - 242 * 2, 'vol'] + hs300.loc[index1 - 242 * 3, 'vol']) / 3  # 前三天量的均值
                new_amount = (hs300.loc[index1 - 242, 'amount'] + hs300.loc[index1 - 242 * 2, 'amount'] + hs300.loc[index1 - 242 * 3, 'amount']) / 3  # 前三天量的均值
                index1 = index1 + 1
                new_time = afternoon_start_time + (i + 1) * timedelta(minutes=1)
                new_data = {'code': code, 'high': new_price, 'low': new_price, 'open': new_price, "avg": new_price,
                            'vol': new_vol, 'amount': new_amount, 'time': new_time}
                max_index = hs300.index.max() + 1
                hs300.loc[max_index] = new_data

#下午开始缺
        else:
            index1 = index
            minutes_difference = int((afternoon_end_time + timedelta(minutes=1) - time).total_seconds() / 60)
            avg_diff_price = (next_data['open'] - row['avg']) / minutes_difference
            new_price = row['avg']
            for i in range(minutes_difference):
                new_price = new_price + avg_diff_price
                new_vol = (hs300.loc[index1 - 242, 'vol'] + hs300.loc[index1 - 242 * 2, 'vol'] + hs300.loc[
                    index1 - 242 * 3, 'vol']) / 3  # 前三天量的均值
                new_amount = (hs300.loc[index1 - 242, 'amount'] + hs300.loc[index1 - 242 * 2, 'amount'] + hs300.loc[
                    index1 - 242 * 3, 'amount']) / 3  # 前三天量的均值
                index1 = index1 + 1
                new_time = time + (i + 1) * timedelta(minutes=1)
                new_data = {'code': code, 'high': new_price, 'low': new_price, 'open': new_price, "avg": new_price,
                            'vol': new_vol, 'amount': new_amount, 'time': new_time}
                max_index = hs300.index.max() + 1
                hs300.loc[max_index] = new_data



#补缺头数据
for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    filtered_data = hs300[hs300['time'].dt.strftime('%Y-%m-%d') == date_str]
    if filtered_data.shape[0] == 0: continue
    morning_start_time = datetime.strptime(date_str + ' ' + "09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end_time = datetime.strptime(date_str + ' ' + "11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start_time = datetime.strptime(date_str + ' ' + "13:00:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end_time = datetime.strptime(date_str + ' ' + "15:00:00", "%Y-%m-%d %H:%M:%S")
    #上午缺头
    for index, row in filtered_data.iterrows():
        time = row['time']
        code = row['code']
        if index == hs300.index.min(): continue
        before_data = hs300.loc[index - 1]
        before_time = before_data['time']
        if time.strftime('%Y-%m-%d') != before_time.strftime('%Y-%m-%d') and time > morning_start_time:
            minutes_difference = int((time-morning_start_time).total_seconds() / 60)
            avg_diff_price = (row['avg']-before_data['open']) / minutes_difference
            print(avg_diff_price)
            new_price = row['avg']
            index1 = index
            for i in range(minutes_difference):
                new_price = new_price - avg_diff_price
                new_vol = (hs300.loc[index1 - 242, 'vol'] + hs300.loc[index1 - 242 * 2, 'vol'] + hs300.loc[
                    index1 - 242 * 3, 'vol']) / 3  # 前三天量的均值
                new_amount = (hs300.loc[index1 - 242, 'amount'] + hs300.loc[index1 - 242 * 2, 'amount'] + hs300.loc[
                    index1 - 242 * 3, 'amount']) / 3  # 前三天量的均值
                index1 = index1 - 1
                new_time = time - (i + 1) * timedelta(minutes=1)
                new_data = {'code': code, 'high': new_price, 'low': new_price, 'open': new_price, "avg": new_price,
                            'vol': new_vol, 'amount': new_amount, 'time': new_time}
                max_index = hs300.index.max() + 1
                hs300.loc[max_index] = new_data



#下午缺头
for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    filtered_data = hs300[hs300['time'].dt.strftime('%Y-%m-%d') == date_str]
    if filtered_data.shape[0] == 0: continue
    morning_start_time = datetime.strptime(date_str + ' ' + "09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end_time = datetime.strptime(date_str + ' ' + "11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start_time = datetime.strptime(date_str + ' ' + "13:00:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end_time = datetime.strptime(date_str + ' ' + "15:00:00", "%Y-%m-%d %H:%M:%S")

    for index, row in filtered_data.iterrows():
        time = row['time']
        code = row['code']
        if index == hs300.index.min(): continue
        before_data = hs300.loc[index - 1]
        before_time = before_data['time']
        if before_time<afternoon_start_time and time > afternoon_start_time+timedelta(minutes=1):
            minutes_difference = int((time-afternoon_start_time-timedelta(minutes=1)).total_seconds() / 60)
            avg_diff_price = (row['avg']-before_data['open']) / minutes_difference
            new_price = row['avg']
            index1 = index
            for i in range(minutes_difference):
                new_price = new_price - avg_diff_price
                new_vol = (hs300.loc[index1 - 242, 'vol'] + hs300.loc[index1 - 242 * 2, 'vol'] + hs300.loc[
                    index1 - 242 * 3, 'vol']) / 3  # 前三天量的均值
                new_amount = (hs300.loc[index1 - 242, 'amount'] + hs300.loc[index1 - 242 * 2, 'amount'] + hs300.loc[
                    index1 - 242 * 3, 'amount']) / 3  # 前三天量的均值
                index1 = index1 - 1
                new_time = time - (i + 1) * timedelta(minutes=1)
                new_data = {'code': code, 'high': new_price, 'low': new_price, 'open': new_price, "avg": new_price,
                            'vol': new_vol, 'amount': new_amount, 'time': new_time}
                max_index = hs300.index.max() + 1
                hs300.loc[max_index] = new_data



grouped_data = hs300.groupby(hs300['time'].dt.date).count()
hs300=hs300.sort_values(by='time')
hs300=hs300.reset_index(drop=True)
hs300.to_csv('Data/B000300/'+file_name,index=False)