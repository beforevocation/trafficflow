'''
Data preprocessing.
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocess_data(train, test, lags):
    '''
    Reshape and split train\test data.
    :param train: String, name of .csv train file.
    :param test: String, name of .csv test file.
    :param lags: integer, time lag.
    :return: 
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    '''

    attr = 'VEH_TOTAL'
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)


    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))

    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1))
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1))
    print(len(flow1))
    print(len(flow2))

    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    np.reshape(X_test, (X_test.shape[0], X_test.shape[1]), 1)
    y_test = test[:, -1]


    return X_train, y_train, X_test, y_test, scaler
