import numpy as np
import pandas as pd


def transform_data(my_data_predict):
    for lag in range(0, 31):
        my_data_predict[f'lag_{lag}'] = my_data_predict['close'].shift(lag)

    # test = my_data_predict[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4']]
    test = my_data_predict[[f'lag_{x}' for x in range(0, 31)]]
    my_date = my_data_predict.index[-1]
    mask = (test.index == my_date)
    test = test.loc[mask]

    # Sin / Cos for the date
    test['date'] = test.index
    test['year'] = pd.DatetimeIndex(test['date']).year
    test['eof'] = pd.to_datetime(test['year'].astype(str) + '-' + str(12) + '-' + str(31))
    test['sin'] = np.sin((2 * np.pi * test['date'].dt.dayofyear) / test['eof'].dt.dayofyear.astype(float))
    test['cos'] = np.cos((2 * np.pi * test['date'].dt.dayofyear) / test['eof'].dt.dayofyear.astype(float))
    test = test.drop(test[['eof', 'year', 'date']], axis=1)
    # print(test)

    return test


# def transform_data(my_data_predict, my_date):
#     for lag in range(0, 5):
#         my_data_predict[f'lag_{lag}'] = my_data_predict['close'].shift(lag)
#
#     test = my_data_predict[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4']]
#     # mask = (test.index == '2022-10-13')
#     mask = (test.index == my_date)
#     test = test.loc[mask]
#     return test

