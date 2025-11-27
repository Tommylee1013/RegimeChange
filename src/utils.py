import numpy as np
import pandas as pd

def ewm_downside_deviation(returns, halflife=10):
    """지수가중 하방편차 계산"""
    negative_returns = returns.where(returns < 0, 0)
    ewm_var = negative_returns.ewm(halflife=halflife).var()
    return np.sqrt(ewm_var)

def ewm_sortino_ratio(returns, halflife_return, halflife_dd=10):
    """지수가중 소르티노 비율 계산"""
    ewm_mean = returns.ewm(halflife=halflife_return).mean()
    ewm_dd = ewm_downside_deviation(returns, halflife=halflife_dd)
    return ewm_mean / ewm_dd