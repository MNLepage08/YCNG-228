import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import scipy.cluster.hierarchy as sch
import pickle


from src.business_logic.process import load_model_in_bucket


def tagger(row):
    if row['next'] < row['lag_0']:
        return 'Sell'
    else:
        return 'Buy'


# Function to calculate the lags of close price
def lags(my_combined_data, my_sp):
    all_lag = pd.DataFrame()
    combined = my_combined_data

    for ticker in my_sp:
        mask = (combined['ticker'] == ticker)
        ticker = combined.loc[mask]

        for lag in range(0, 31):
            ticker[f'lag_{lag}'] = ticker['close'].shift(lag)
        all_lag = all_lag.append(ticker)

    # Keep only interested columns
    all_lag = all_lag.drop(all_lag[['open', 'high', 'low', 'adjclose', 'volume']], axis=1)
    all_lag['date'] = all_lag.index
    all_lag = all_lag.dropna()

    # print(all_lag)
    return all_lag


def train_baseline_model(my_combined_data, my_sp, my_date):

    # Date of the train model
    date_object = datetime.strptime(my_date, '%Y-%m-%d').date()
    date_object = date_object.replace(day=1)

    # The name of the model based on the date
    version_model = 'model_' + str(date_object.year) + '_' + str(date_object.month) + '_v2.pkl'

    # Calculate the lag of the close price
    all_data = lags(my_combined_data, my_sp)

    # # Calculate cluster
    # df = pd.DataFrame()
    # sector = all_data['sector'].unique().tolist()
    #
    # for s in sector:
    #
    #     # list of ticker for each sector
    #     s = all_data.loc[all_data['sector'].isin([s])]
    #     t = s.loc[s['date'] == '2022-06-01']
    #     s = t.ticker.values.tolist()

        # # correlation
        # df_res = pd.DataFrame()
        # for ticker in s:
        #     # df_tmp = si.get_data(ticker, start_date='06-01-2022')
        #     df_res[ticker] = t['close']
        #     corr = df_res.corr()
        #
        # # cluster
        # c = list()
        # cc = list()
        #
        # # retrieve clusters using fcluster
        # d = sch.distance.pdist(corr)
        # L = sch.linkage(d, method='complete')
        # # 0.2 can be modified to retrieve more stringent or relaxed clusters
        # clusters = sch.fcluster(L, 0.2 * d.max(), 'distance')

        # # clusters indicices correspond to incides of original df
        # for i, cluster in enumerate(clusters):
        #     c = corr.index[i], cluster
        #     cc.append(c)
        # # print(cc)
        #
        # df = df.append(pd.DataFrame(cc, columns=['ticker', 'cluster']))

    # all_data = all_data.merge(df, on='ticker')
    # all_data['sc_nbr'] = all_data['sector'] + '-' + all_data['cluster'].astype(str)
    #
    # spn = all_data['sc_nbr'].unique().tolist()
    # id_sp = dict(zip(spn, range(len(spn))))
    # df = pd.DataFrame(id_sp.items())
    # df.rename(columns={0: 'sc_nbr', 1: 'number'}, inplace=True)
    #
    # all_data = all_data.merge(df, on='sc_nbr')
    # print(all_data)

    # Calculate Buy/Sell
    all_data['next'] = all_data['close'].shift(-1)
    all_data['out'] = all_data.apply(tagger, axis=1)

    # Sin / Cos for the date
    all_data['year'] = pd.DatetimeIndex(all_data['date']).year
    all_data['eof'] = pd.to_datetime(all_data['year'].astype(str) + '-' + str(12) + '-' + str(31))
    all_data['sin'] = np.sin((2 * np.pi * all_data['date'].dt.dayofyear) / all_data['eof'].dt.dayofyear.astype(float))
    all_data['cos'] = np.cos((2 * np.pi * all_data['date'].dt.dayofyear) / all_data['eof'].dt.dayofyear.astype(float))
    all_data = all_data.drop(all_data[['eof', 'year']], axis=1)

    # Take only data before the date in the app
    mask = (all_data['date'] <= str(date_object))
    all_data = all_data.loc[mask]

    list_date = all_data.loc[(all_data['date'] <= str(date_object))]
    list_date = list_date['date']
    list_date = list_date.unique()
    list_date = np.sort(list_date)

    train_date = all_data.loc[(all_data['date'] < str(date_object - relativedelta(months=3)))]
    train_date = train_date['date']
    train_date = train_date.unique()
    train_date = np.sort(train_date)

    prediction = []
    actual = []

    # Train model for each day (during 3 months)
    for i in range(len(train_date), len(list_date)):
        a = list_date[i]
        train = all_data.loc[all_data['date'] < a]
        test = all_data.loc[all_data['date'] == a]

        train_x = train.drop(train[['out', 'date', 'close', 'ticker', 'next']], axis=1)
        train_y = train['out']
        test_x = test.drop(test[['out', 'date', 'close', 'ticker', 'next']], axis=1)
        test_y = test['out']

        # model_lr = LogisticRegression()
        model_lr = lgb.LGBMClassifier(n_estimators=95, num_leaves=64, max_depth=5, learning_rate=0.1, random_state=1000, n_jobs=-1)
        model_lr.fit(train_x, train_y)

        predict = model_lr.predict(test_x)
        prediction.append(predict)
        actual.append(test_y.values)

    # Score of the baseline model
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)

    all_actual = np.concatenate([actual[x] for x in range(len(actual))])
    all_prediction = np.concatenate([prediction[x] for x in range(len(prediction))])
    score = balanced_accuracy_score(all_actual, all_prediction)
    score = str(score)

    # Load the file locally
    filename = version_model
    pickle.dump(model_lr, open(filename, 'wb'))

    # Load the file in GCP
    load_model_in_bucket(my_date)

    return score

# def train_baseline_model(my_combined_data, my_sp, my_date):
#     # 2022 - 10 - 01
#
#     date_object = datetime.strptime(my_date, '%Y-%m-%d').date()
#     date_object = date_object.replace(day=1)
#     version_model = 'model_' + str(date_object.year) + '_' + str(date_object.month) + '_v2' + '.pkl'
#     combined = my_combined_data
#
#     # Create lag of close price
#     all_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker',
#                                      'lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4'])
#     for ticker in my_sp:
#       mask = (combined['ticker'] == ticker)
#       ticker = combined.loc[mask]
#
#       for lag in range(0, 5):
#           ticker[f'lag_{lag}'] = ticker['close'].shift(lag)
#       all_data = all_data.append(ticker)
#
#     # Calculate Buy/Sell
#     all_data['next'] = all_data['close'].shift(-1)
#     all_data['out'] = all_data.apply(tagger, axis=1)
#     all_data = all_data[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'out']]
#     all_data['date'] = all_data.index
#     all_data = all_data[all_data['lag_4'].notnull()]
#
#     # Take only data before date enter in the app
#     mask = (all_data['date'] <= str(date_object))
#     all_data = all_data.loc[mask]
#
#     List_Date = all_data.loc[(all_data['date'] <= str(date_object))]
#     List_Date = List_Date['date']
#     List_Date = List_Date.unique()
#     List_Date = np.sort(List_Date)
#
#     # Train_Date = all_data.loc[(all_data['date'] < '2022-06-01')]
#     # Train_Date = all_data.loc[(all_data['date'] < str(date_object - timedelta(days=90)))]
#     Train_Date = all_data.loc[(all_data['date'] < str(date_object - relativedelta(months=3)))]
#
#     Train_Date = Train_Date['date']
#     Train_Date = Train_Date.unique()
#     Train_Date = np.sort(Train_Date)
#
#     prediction = []
#     actual = []
#
#     # Train model for each day (during 3 months)
#     for i in range(len(Train_Date), len(List_Date)):
#         a = List_Date[i]
#         train = all_data.loc[all_data['date'] < a]
#         test = all_data.loc[all_data['date'] == a]
#
#         train = train.drop('date', axis=1)
#         test = test.drop('date', axis=1)
#
#         trainX = train[[f'lag_{lag}' for lag in range(0, 5)]]
#         trainY = train['out']
#         testX = test[[f'lag_{lag}' for lag in range(0,5)]]
#         testY = test['out']
#
#         model = LogisticRegression()
#         model.fit(trainX, trainY)
#
#         pred_b = model.predict(testX)
#         prediction.append(pred_b)
#         actual.append(testY.values)
#
#     # Score of the baseline model
#     prediction = pd.Series(prediction)
#     actual = pd.Series(actual)
#
#     all_actual = np.concatenate([actual[x] for x in range(len(actual))])
#     all_prediction = np.concatenate([prediction[x] for x in range(len(prediction))])
#     score = balanced_accuracy_score(all_actual, all_prediction)
#     score = str(score)
#
#     # Load the file locally
#     #filename = 'my_model_v6.pkl'
#     filename = version_model
#     pickle.dump(model, open(filename, 'wb'))
#
#     # Load the file in GCP
#     load_model_in_bucket(my_date)
#
#     return score
#
#
#
#
