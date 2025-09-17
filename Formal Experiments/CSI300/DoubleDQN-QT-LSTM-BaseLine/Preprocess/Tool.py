from collections import Iterable
import numpy as np
#https://cloud.tencent.com/developer/article/1518357
#对于hust指数来说，高于0.87的视为一段趋势，当一段上涨趋势终结时，其最高价会形成一个顶部，进入横盘阶段时，最低价形成支撑，对于下降趋势则反过来
def Hurst(ts):
    '''
    Parameters
    ----------
    ts : Iterable Object.
        A time series or a list.

    Raises
    ------
    ValueError
        If input ts is not iterable then raise error.

    Returns
    -------
    H : Float
        The Hurst-index of this series.
    '''
    if not isinstance(ts, Iterable):
        raise ValueError("This sequence is not iterable !")
    ts = np.array(ts)
    # N is use for storge the length sequence
    N, RS, n = [], [], len(ts)
    while (True):
        N.append(n)
        # Calculate the average value of the series
        m = np.mean(ts)
        # Construct mean adjustment sequence
        mean_adj = ts - m
        # Construct cumulative deviation sequence
        cumulative_dvi = np.cumsum(mean_adj)
        # Calculate sequence range
        srange = max(cumulative_dvi) - min(cumulative_dvi)
        # Calculate the unbiased standard deviation of this sequence
        unbiased_std_dvi = np.std(ts)
        # Calculate the rescaled range of this sequence under n length
        RS.append(srange / unbiased_std_dvi)
        # While n < 4 then break
        if n < 4:
            break
        # Rebuild this sequence by half length
        ts, n = HalfSeries(ts, n)
    # Get Hurst-index by fit log(RS)~log(n)
    H = np.polyfit(np.log10(N), np.log10(RS), 1)[0]
    return H

def HalfSeries(s, n):
    '''
    if length(X) is odd:
        X <- {(X1 + X2) / 2, ..., (Xn-2 + Xn-1) / 2, Xn}
        n <- (n - 1) / 2
    else:
        X <- {(X1 + X2) / 2, ..., (Xn-1 + Xn) / 2}
        n <- n / 2
    return X, n
    '''
    X = []
    for i in range(0, len(s) - 1, 2):
        X.append((s[i] + s[i + 1]) / 2)
    # if length(s) is odd
    if len(s) % 2 != 0:
        X.append(s[-1])
        n = (n - 1) // 2
    else:
        n = n // 2
    return [np.array(X), n]

def stdOfReturn(ts):
    length=len(ts)
    returnRate=[]
    for i in range(length-1):
        returnRate.append(ts[i+1]-ts[i])
    return np.std(returnRate)