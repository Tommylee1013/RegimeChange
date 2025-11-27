import pandas as pd
import numpy as np

from tqdm import tqdm
from src.model import StatisticalJumpModel

def rolling_monthly_states(
        features: pd.DataFrame,
        train_window_years: int = 10,
        horizon_months: int = 1,
        n_states: int = 2,
        jump_penalty: float = 50.0,
        max_iter: int = 10,
        random_state: int = 42,
        dist_method : str = 'median'
    ) -> pd.Series:
    """
    과거 train_window_years년 데이터를 이용해 1개월(horizon_months) 동안의 상태를
    rolling으로 추정하는 함수.

    Parameters
    ----------
    features : pd.DataFrame
        상태 추정에 사용할 feature (예: SPX return, excess return).
        DatetimeIndex 를 가지고 있어야 함.
    train_window_years : int
        각 기준 시점에서 뒤로 몇 년을 훈련 데이터로 사용할지 (기본 10년).
    horizon_months : int
        앞으로 몇 개월을 예측(상태 추정)할지 (기본 1개월).
    n_states, jump_penalty, max_iter, random_state :
        StatisticalJumpModel 의 하이퍼파라미터.

    Returns
    -------
    all_states : pd.Series
        rolling으로 추정된 상태 시계열.
        index = features.index 의 부분집합, name='state'
    """
    features = features.sort_index()
    idx = features.index

    month_start_idx = features.resample("MS").first().dropna().index

    min_start_date = idx[0] + pd.DateOffset(years=train_window_years)
    month_start_idx = month_start_idx[month_start_idx >= min_start_date]

    results = []

    for start_test in tqdm(month_start_idx):

        # training periods
        train_start = start_test - pd.DateOffset(years=train_window_years)
        train_end = start_test - pd.Timedelta(days=1)
        train_X = features.loc[train_start:train_end]
        if len(train_X) < 60:
            continue

        test_end = start_test + pd.DateOffset(months=horizon_months) - pd.Timedelta(days=1)
        test_X = features.loc[start_test:test_end]
        if test_X.empty:
            continue

        model = StatisticalJumpModel(
            n_states=n_states,
            jump_penalty=jump_penalty,
            max_iter=max_iter,
            random_state=random_state,
            dist_method=dist_method
        )
        model.fit(train_X)

        states = model.filter_sequence(test_X) # Series
        prob = model.predict_proba(test_X) # DataFrame
        df_out = pd.concat([states, prob], axis=1)

        results.append(df_out)

    if not results:
        return pd.DataFrame(columns=["state"] + [f"state_{i}_prob" for i in range(n_states)])

    final_df = pd.concat(results).sort_index()
    return final_df