from typing import List

import numpy as np
import pandas as pd


def get_mean_log(logs: List[pd.DataFrame]):
    base_df = logs[0].copy()
    for _df in logs[1:]:
        base_df = base_df.add(_df, fill_value=0)
    return base_df / len(logs)


def get_std_log(logs: List[pd.DataFrame]):
    mean_df = get_mean_log(logs)
    base_df = logs[0].copy() * 0
    for _df in logs:
        base_df += np.power((_df - mean_df), 2)
    return np.sqrt(base_df / len(logs))
