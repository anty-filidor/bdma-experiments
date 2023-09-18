from typing import List
import network_diffusion as nd
import networkx as nx
import pandas as pd


def get_mean_log(logs: List[pd.DataFrame]):
    base_df = logs[0].copy()
    for _df in logs[1:]:
        base_df = base_df.add(_df, fill_value=0)
    return base_df / len(logs)

def get_std_log(logs: List[pd.DataFrame]):
    ...

