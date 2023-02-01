import pandas as pd
import numpy as np
from typing import List

  
def make_price_threshold(df: pd.DataFrame) -> List[float]:
    """
    Make the thresholds for the price based on a dataframe.
    :param df: Dataframe on which the threshold is based.
    :returns a List with the thresholds
    """
    price_dict = df.groupby('Year')['Price (Euros)'].quantile([0.25, 0.5, 0.75, 1]).to_dict()
    thresholds = {}

    for key, value in price_dict.items():
        if key[0] in thresholds:
            thresholds[key[0]].append(value)
        else:
            thresholds[key[0]] = [value]

    return list(np.median([value for key, value in thresholds.items()], axis=0))[:-1]


def make_water_threshold() -> List[int]:
    """
    Hardcoded water threshold.
    :return: threshold list
    """
    return [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
