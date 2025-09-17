# 基于Black - Scholes 公式的期权定价公式
#https://blog.csdn.net/ndhtou222/article/details/109831509
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
import calendar
import pandas as pd
from datetime import datetime, timedelta
import time

class BSM():
    def greeks(self,CP, S, X, sigma, T, r, b):  # 计算greeks的函数
        """
        Parameters
        ----------
        CP：看涨或看跌"C"or"P"
        S : 标的价格.
        X : 行权价格.
        sigma :波动率.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本，当b = r 时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        Returns
        -------
        返回欧式期权的估值和希腊字母
        """
        d1 = (np.log(S / X) + (b + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))  # 求d1
        d2 = d1 - sigma * np.sqrt(T)  # 求d2

        if CP == "C":
            option_value = S * exp((b - r) * T) * norm.cdf(d1) - X * exp(-r * T) * norm.cdf(d2)  # 计算期权价值
            delta = exp((b - r) * T) * norm.cdf(d1)
            gamma = exp((b - r) * T) * norm.pdf(d1) / (S * sigma * T ** 0.5)  # 注意是pdf，概率密度函数
            vega = S * exp((b - r) * T) * norm.pdf(d1) * T ** 0.5  # 计算vega
            theta = -exp((b - r) * T) * S * norm.pdf(d1) * sigma / (2 * T ** 0.5) - r * X * exp(-r * T) * norm.cdf(
                d2) - (b - r) * S * exp((b - r) * T) * norm.cdf(d1)
            speed = -(gamma / S) * ((d1 / (sigma * np.sqrt(T))) + 1)
            if b != 0:  # rho比较特别，b是否为0会影响求导结果的形式
                rho = X * T * exp(-r * T) * norm.cdf(d2)
            else:
                rho = -exp(-r * T) * (S * norm.cdf(d1) - X * norm.cdf(d2))

        else:
            option_value = X * exp(-r * T) * norm.cdf(-d2) - S * exp((b - r) * T) * norm.cdf(-d1)
            delta = -exp((b - r) * T) * norm.cdf(-d1)
            gamma = exp((b - r) * T) * norm.pdf(d1) / (S * sigma * T ** 0.5)  # 跟看涨其实一样，不过还是先写在这里
            vega = S * exp((b - r) * T) * norm.pdf(d1) * T ** 0.5  # #跟看涨其实一样，不过还是先写在这里
            theta = -exp((b - r) * T) * S * norm.pdf(d1) * sigma / (2 * T ** 0.5) + r * X * exp(-r * T) * norm.cdf(
                -d2) + (b - r) * S * exp((b - r) * T) * norm.cdf(-d1)
            speed = -(gamma / S) * ((d1 / (sigma * np.sqrt(T))) + 1)
            if b != 0:  # rho比较特别，b是否为0会影响求导结果的形式
                rho = -X * T * exp(-r * T) * norm.cdf(-d2)
            else:
                rho = -exp(-r * T) * (X * norm.cdf(-d2) - S * norm.cdf(-d1))
        #  写成函数时要有个返回，这里直接把整个写成字典一次性输出。
        greeks = {"option_value": option_value, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
                  "rho": rho, "speed": speed}
        return greeks


    def get_expiration_time(self,time):
        cal = calendar.monthcalendar(time.year, time.month)
        if cal[0][4]==0:
            friday=3
        else:
            friday=2

        time1=pd.to_datetime(str(time.year)+'-'+str(time.month)+'-'+str(cal[friday][4]))#当月的第三个礼拜五，第一个维度是第几个礼拜，第二个维度是礼拜几
        ExpirationThisMonth=time1-time


        Time2month=time.month
        Time2year = time.year
        if time.month==12:
            cal = calendar.monthcalendar(time.year+1, 1)
            Time2month=1
            Time2year = time.year+1
        else:
            cal = calendar.monthcalendar(time.year, time.month+1)
            Time2month=Time2month+1
        if cal[0][4]==0:
            friday=3
        else:
            friday=2
        time2 = pd.to_datetime(str(Time2year) + '-' + str(Time2month) + '-' + str(cal[friday][4]))  # 次月的第三个礼拜五，第一个维度是第几个礼拜，第二个维度是礼拜几

        if ExpirationThisMonth.days>15:
            return time1
        else:
            return time2





if __name__ == '__main__':
    bsm=BSM()
    #bsm.get_expiration_time(pd.to_datetime('2023-12-22 11:16:00'))
    print(bsm.greeks(CP='P',S=3.68,X=3.6,sigma=0.14,T=36/360,r=0.021,b=0))







