import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as pdr

from src.utils import *
from src.simulation import rolling_monthly_states

warnings.filterwarnings('ignore')

DIST_METHOD : str = 'huber'
RANDOM_STATE : int = 42
TRAIN_YEARS : int = 10
PREDICT_MONTH : int = 1
JUMP_PENALTY : float = 0.2
NUM_OF_STATES : int = 2

if __name__ == '__main__':
    # S&P500 price
    spx = yf.download(
        '^GSPC',
        start='1900-01-01',
        progress=False,
        interval='1d',
        multi_level_index=False,
        auto_adjust=False
    )
    # T bill 3 Month, 무위험 이자율을 계산하기 위함
    tbill = pdr.get_data_fred(
        'DGS3MO',
        start='1900-01-01'
    ).ffill()
    # 국채 2년물과 10년물, 스프레드를 계산하기 위함
    tnote2 = pdr.get_data_fred(
        'DGS2',
        start='1900-01-01'
    ).ffill()
    tnote10 = pdr.get_data_fred(
        'DGS10',
        start='1900-01-01'
    ).ffill()

    spread = (tnote2['DGS2'] - tnote10['DGS10']).dropna()

    price = pd.concat([
        spx['Close'],
        spx['Close'].pct_change(),
        tbill * 0.01,
        tbill * 0.01 / 252,
        spread
    ], axis=1).dropna()
    price.columns = ['SPX', 'SPX return', 'Tbill 3Month', 'risk free', 'spread']

    # 초과수익률 계산
    price['excess return'] = price['SPX return'] - price['risk free']

    downside_dev = ewm_downside_deviation(price['excess return'], halflife=10)
    sortino_20 = ewm_sortino_ratio(price['excess return'], halflife_return=20, halflife_dd=10)
    sortino_60 = ewm_sortino_ratio(price['excess return'], halflife_return=60, halflife_dd=10)
    ewm_std_20 = price['excess return'].ewm(20).std()
    ewm_std_60 = price['excess return'].ewm(60).std()

    feature_matrix = pd.concat(
        [
            downside_dev,
            sortino_20,
            sortino_60,
            spread * 0.01,
            ewm_std_20,
            ewm_std_60,
        ], axis=1)
    feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_matrix.columns = [
        'downside deviation',
        'sortino 20',
        'sortino 60',
        'spread',
        'std 20',
        'std 60',
    ]

    states_rolling = rolling_monthly_states(
        features=feature_matrix.dropna(),
        train_window_years = TRAIN_YEARS,
        horizon_months = PREDICT_MONTH,
        n_states = NUM_OF_STATES,
        jump_penalty = JUMP_PENALTY,
        max_iter = 100,
        random_state = RANDOM_STATE,
        dist_method = DIST_METHOD
    )

    sell_signal = states_rolling.loc[:, 'state_0_prob']

    sell_signal.loc['2018':].plot()
    price.loc['2018':, 'SPX'].plot(secondary_y=True)
    plt.title('regime change probability')
    plt.show()