import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score


def load_data(filename, target_col, shift=1, keep_last=False):
    """
    Given a data source, loads appropriate csv file.
    """

    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    df.index = df.index.rename('date')
    df = df.loc[~df.index.duplicated(keep='last')]
    labels = df[target_col].shift(-shift)
    price = df[target_col].rename('price')

    if keep_last:
        return df, labels, price
    else:
        index = pd.notnull(labels)
        return df[index], labels[index], price[index]


def fetch_crypto_rate(filename, from_currency, to_currency, start, end, freq):
    """
    Given currencies and start/end dates, as well as frequency, gets exchange rates from Poloniex API.
    """

    base_url = "https://poloniex.com/public?command=returnChartData"
    base_url += "&currencyPair=" + from_currency + "_" + to_currency
    data = []

    start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    tmp1 = start
    if end - start < timedelta(weeks=12):
        tmp2 = end
    else:
        tmp2 = start + timedelta(weeks=12)
    while tmp2 <= end:
        x1, x2 = datetime.timestamp(tmp1), datetime.timestamp(tmp2)
        main_url = base_url + "&start=" + str(x1) + "&end=" + str(x2) + "&period=" + str(freq * 60)
        print('Fetching:', main_url)
        res = requests.get(main_url).json()
        if res[0]['date'] != 0:
            data += requests.get(main_url).json()

        tmp1, tmp2 = tmp1 + timedelta(weeks=12), tmp2 + timedelta(weeks=12)
        if tmp1 < end < tmp2:
            tmp2 = end

    df = pd.DataFrame.from_dict(data).set_index('date')
    df.index = pd.to_datetime(df.index, unit='s') #.tz_localize('UTC').tz_convert('Europe/Paris')
    df.to_csv(filename, encoding='utf-8')


def fetch_currency_rate(filename, from_currency, to_currency, freq, api_key):
    """
    Given a currency pair, gets all possible historic data from Alphavantage.
    """

    base_url = r"https://www.alphavantage.co/query?function=FX_INTRADAY"
    base_url += "&interval=" + str(freq) + "min&outputsize=full" + "&apikey=" + api_key

    url = base_url + "&from_symbol=" + from_currency + "&to_symbol=" + to_currency
    print('Fetching:', url)
    data = requests.get(url).json()["Time Series FX (" + str(freq) + "min)"]
    df1 = pd.DataFrame.from_dict(data, orient='index')
    df1.columns = [(from_currency + to_currency + x[3:]) for x in df1.columns]

    url = base_url + "&from_symbol=" + to_currency + "&to_symbol=" + from_currency
    print('Fetching:', url)
    data = requests.get(url).json()["Time Series FX (" + str(freq) + "min)"]
    df2 = pd.DataFrame.from_dict(data, orient='index')
    df2.columns = [(to_currency + from_currency + x[3:]) for x in df2.columns]

    if not df1.index.equals(df2.index):
        print('Warning: index not aligned btw exchange rate and reverse exchange rate.')

    df = pd.concat([df1, df2], axis=1, sort=True).dropna()

    if os.path.exists(filename):
        old = pd.read_csv(filename, encoding='utf-8', index_col=0)
        df = pd.concat([old, df]).drop_duplicates(keep='last')

    df.to_csv(filename, encoding='utf-8')
    print('New available EUR-GBP data shape:', df.shape)


def compute_metrics(y_true, y_pred):
    """
    Given true labels and predictions, outputs a set of performance metrics for regression task.
    """

    are = np.abs(y_pred - y_true) / np.abs(y_true)
    r2 = r2_score(y_true, y_pred)
    me = max_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pred_shift = np.array(y_pred[:-1] > y_true[1:]).astype(int)
    true_shift = np.array(y_true[:-1] > y_true[1:]).astype(int)

    return {'max_abs_rel_error':    are.max(),
            'mean_abs_rel_error':   are.mean(),
            'mean_absolute_error':  mae,
            'max_error':            me,
            'mean_squared_error':   mse,
            'r2':                   r2,
            'change_accuracy':           (pred_shift == true_shift).mean()
            }


def evaluate(portfolio, rate):
    return portfolio[0] + (portfolio[1] * rate)
