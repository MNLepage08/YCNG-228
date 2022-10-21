import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle


from src.business_logic.process import load_model_in_bucket


def tagger(row):
    if row['next'] < row['lag_0']:
        return 'Sell'
    else:
        return 'Buy'


def train_baseline_model(my_combined_data, my_sp, my_date):
    # 2022 - 10 - 01

    date_object = datetime.strptime(my_date, '%Y-%m-%d').date()
    date_object = date_object.replace(day=1)
    version_model = 'model_' + str(date_object.year) + '_' + str(date_object.month) + '.pkl'
    combined = my_combined_data

    # Create lag of close price
    all_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker',
                                     'lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4'])
    for ticker in my_sp:
      mask = (combined['ticker'] == ticker)
      ticker = combined.loc[mask]

      for lag in range(0, 5):
          ticker[f'lag_{lag}'] = ticker['close'].shift(lag)
      all_data = all_data.append(ticker)

    # Calculate Buy/Sell
    all_data['next'] = all_data['close'].shift(-1)
    all_data['out'] = all_data.apply(tagger, axis=1)
    all_data = all_data[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'out']]
    all_data['date'] = all_data.index
    all_data = all_data[all_data['lag_4'].notnull()]

    # Take only data before date enter in the app
    mask = (all_data['date'] <= str(date_object))
    all_data = all_data.loc[mask]

    List_Date = all_data.loc[(all_data['date'] <= str(date_object))]
    List_Date = List_Date['date']
    List_Date = List_Date.unique()
    List_Date = np.sort(List_Date)

    # Train_Date = all_data.loc[(all_data['date'] < '2022-06-01')]
    # Train_Date = all_data.loc[(all_data['date'] < str(date_object - timedelta(days=90)))]
    Train_Date = all_data.loc[(all_data['date'] < str(date_object - relativedelta(months=3)))]

    Train_Date = Train_Date['date']
    Train_Date = Train_Date.unique()
    Train_Date = np.sort(Train_Date)

    prediction = []
    actual = []

    # Train model for each day (during 3 months)
    for i in range(len(Train_Date), len(List_Date)):
        a = List_Date[i]
        train = all_data.loc[all_data['date'] < a]
        test = all_data.loc[all_data['date'] == a]

        train = train.drop('date', axis=1)
        test = test.drop('date', axis=1)

        trainX = train[[f'lag_{lag}' for lag in range(0, 5)]]
        trainY = train['out']
        testX = test[[f'lag_{lag}' for lag in range(0,5)]]
        testY = test['out']

        model = LogisticRegression()
        model.fit(trainX, trainY)

        pred_b = model.predict(testX)
        prediction.append(pred_b)
        actual.append(testY.values)

    # Score of the baseline model
    prediction = pd.Series(prediction)
    actual = pd.Series(actual)

    all_actual = np.concatenate([actual[x] for x in range(len(actual))])
    all_prediction = np.concatenate([prediction[x] for x in range(len(prediction))])
    score = balanced_accuracy_score(all_actual, all_prediction)
    score = str(score)

    # Load the file locally
    #filename = 'my_model_v6.pkl'
    filename = version_model
    pickle.dump(model, open(filename, 'wb'))

    # Load the file in GCP
    load_model_in_bucket(my_date)

    return score




