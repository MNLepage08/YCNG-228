import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import pandas as pd
from yahoo_fin import stock_info as si
import numpy as np

def tagger(row):
    if row['next'] < row['lag_0']:
        return 'Sell'
    else:
        return 'Buy'

def create_features(df_stock):
    combined = df_stock.copy()
    all_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker', 'lag_0', 'lag_1',
                                     'lag_2', 'lag_3', 'lag_4'])
    sp = si.tickers_sp500()
    #sp = df_ticker.copy()
    for ticker in sp:

        mask = (combined['ticker'] == ticker)
        ticker = combined.loc[mask]

        for lag in range(0, 5):
            ticker[f'lag_{lag}'] = ticker['close'].shift(lag)

        all_data = all_data.append(ticker)
        all_data['next'] = all_data['close'].shift(-1)

    all_data['out'] = all_data.apply(tagger, axis=1)

    all_data = all_data[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'out']]
    all_data['date'] = all_data.index
    all_data = all_data[all_data['lag_4'].notnull()]
    mask = (all_data['date'] <= '2022-09-01')
    all_data = all_data.loc[mask]

    return all_data


def create_X_Y(df_lags):
    # X = df_lags.drop('lags_0', axis=1)
    # Y = df_lags[['lags_0']]

    list_date = df_lags.loc[(df_lags['date'] <= '2022-09-01')]
    list_date = list_date['date']
    list_date = list_date.unique()
    list_date = np.sort(list_date)

    train_date = df_lags.loc[(df_lags['date'] < '2022-06-01')]
    train_date = train_date['date']
    train_date = train_date.unique()
    train_date = np.sort(train_date)

    for i in range(len(train_date), len(list_date)):

        a = list_date[len(train_date)]

        train = df_lags.loc[df_lags['date'] < a]
        #test = df_lags.loc[df_lags['date'] == a]

        train = train.drop('date', axis=1)
        #test = test.drop('date', axis=1)

        train_x = train[[f'lag_{lag}' for lag in range(0, 5)]]
        train_y = train['out']
        testX = test[[f'lag_{lag}' for lag in range(0, 5)]]
        testY = test['out']

    return train_x, train_y



class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LogisticRegression()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features = create_X_Y(df_features)
        self.lr.fit(df_features)
        return self

    def predict(self, X):
        print(X)
        data = self._data_fetcher(X, last=True)
        print(data)
        df_features = create_features(data)
        print(df_features)
        df_features = create_X_Y(df_features)
        predictions = self.lr.predict(df_features)

        return predictions.flatten()[-1]
