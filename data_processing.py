if __name__ == "__main__":
    import torch
    from torch.utils.data.dataset import Subset
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset

    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import pandas_ta as ta
    import itertools
    import numpy as np

    from tqdm.auto import tqdm
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    import math
    from utils import *

    class IndicatorDataset(Dataset):
        def __init__(
            self,
            ticker: str,
            start_date: str,
            end_date: str,
            device,
            dataset: pd.DataFrame = None,
            window_size=11,
        ) -> None:
            self.data = dataset
            self.transform = MinMaxScaler(feature_range=(-1, 1))

            self.device = device
            self.window_size = window_size
            self.ticker = ticker
            self.start_date = start_date
            self.end_date = end_date
            self.data: pd.DataFrame = self._get_data(
                self.ticker, self.start_date, self.end_date
            )  # if dataset == None else dataset
            self.labels: np.ndarray = self._create_labels()[
                int(window_size / 2 + 28) : -int(window_size / 2 + 28)
            ]

        def _get_data(self, ticker: str, start_date: str, end_date: str):
            df = yf.download(ticker, start_date, end_date)
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
                {"kind": "inertia"},
                {"kind": "pgo"},
                {"kind": "roc"},
                {"kind": "cmf"},
                {"kind": "mom"},
                {"kind": "psl"},
            ]
            ta_s = [
                {**indicator, **length[i]}
                for indicator, i in itertools.product(
                    indicators, range(len(indicators))
                )
            ]
            window_size = 11
            MyStrategy = ta.Strategy(name="15x15", ta=ta_s)
            df.ta.strategy(MyStrategy)
            return df

        def _create_labels(self, window_size=11):
            df = self.data
            col_name = "Close"
            total_rows = len(df)
            labels = np.zeros(total_rows)
            labels[:] = np.nan
            print("Calculating labels")
            pbar = tqdm(total=total_rows)

            for row_counter in range(total_rows):
                if row_counter >= self.window_size - 1:
                    window_begin = row_counter - (self.window_size - 1)
                    window_end = row_counter
                    window_middle = int((window_begin + window_end) / 2)
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

        def plot_data(self):
            plt.figure(figsize=(10, 7))
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.plot(self.data["Close"], lw=1, label="Close Price")
            for idx, column in enumerate(self.data, start=6):
                if idx % 15 == 6:
                    plt.plot(self.data[column], lw=1, label=column)
            plt.legend()
            plt.show()

        def _transform_features(self) -> np.ndarray:
            # Scale features:
            scaler = self.transform
            scaler.fit(
                self.data.iloc[
                    int(self.window_size / 2 + 9) : -int(self.window_size / 2), 6:
                ].values
            )
            scaled_features = scaler.transform(
                self.data.iloc[
                    int(self.window_size / 2 + 28) : -int(self.window_size / 2 + 28), 6:
                ].values
            )
            return scaled_features

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tensor:
            squarable_datapoint = self._transform_features()[idx]
            assert is_sqrt(len(squarable_datapoint))
            len_square = int(math.sqrt(len(squarable_datapoint)))
            squared_datapoint = squarable_datapoint
            squared_datapoint = torch.from_numpy(squarable_datapoint).reshape(
                len_square, len_square
            )
            label_position = int(self.labels[idx])
            label = np.zeros(3)
            label[label_position] = 1
            label = torch.from_numpy(label)

            return squared_datapoint.unsqueeze(0).float(), label.float()

    batch_size = 64
    shuffle = True
    drop_last = True
    num_workers = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_test_ratio = 0.8

    full_dataset = IndicatorDataset("GOOGL", "2020-01-01", "2022-04-30", device=device)

    train_size = int(train_test_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    full_dataset[2]
