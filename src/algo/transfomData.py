def transform_data(my_data_predict, my_date):
    for lag in range(0, 5):
        my_data_predict[f'lag_{lag}'] = my_data_predict['close'].shift(lag)

    test = my_data_predict[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4']]
    # mask = (test.index == '2022-10-13')
    mask =(test.index == my_date)
    test = test.loc[mask]
    return test

