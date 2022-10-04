import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta as ta
import itertools
import numpy as np
from tqdm.auto import tqdm


df = yf.download("GOOGL", start="2020-01-01", end="2022-04-30")
df.columns


def ta1515():
    length = [{"length": i} for i in range(6, 20 + 1)]
    indicators = [
        {"kind": "rsi"},
        {"kind": "willr"},
        {"kind": "fwma"},
        {"kind": "ema"},
        {"kind": "sma"},
        {"kind": "hma"},
        {"kind": "tema"},
        {"kind": "cci"},
        {"kind": "cmo"},
        {"kind": "macd"},
        {"kind": "ppo"},
        {"kind": "roc"},
        {"kind": "cmf"},
        {"kind": "dm"},
        {"kind": "psl"},
    ]
    ta = [
        {**indicator, **length[i]}
        for indicator, i in itertools.product(indicators, range(15))
    ]
    return ta


MyStrategy = ta.Strategy(name="15x15", ta=ta1515())
df.ta.strategy(MyStrategy)
df.columns
df.tail()


def create_labels(df, col_name, window_size=11):
    total_rows = len(df)
    labels = np.zeros(total_rows)
    labels[:] = np.nan
    print("Calculating labels")
    pbar = tqdm(total=total_rows)

    for row_counter in range(total_rows):
        if row_counter >= window_size - 1:
            window_begin = row_counter - (window_size - 1)
            window_end = row_counter
            window_middle = (window_begin + window_end) / 2

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin, window_end + 1):
                price = df.iloc[i][col_name]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[window_middle] = 0
            elif min_index == window_middle:
                labels[window_middle] = 1
            else:
                labels[window_middle] = 2

        pbar.update(1)

    pbar.close()
    return labels
