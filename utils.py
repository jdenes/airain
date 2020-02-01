import glob
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, r2_score


def load_data(datatype='crypto'):
    """
    Given a data source in 'crypto', 'long_currency' or 'short-currency', loads appropriate csv file.
    """

    if datatype == 'crypto':
        df = pd.read_csv('./data/csv/dataset_crypto.csv', encoding='utf-8', index_col=0)
        labels = df['weightedAverage'].shift(-1)
    elif datatype == 'short_currency':
        df = pd.read_csv('./data/csv/dataset.csv', encoding='utf-8', index_col=0)
        labels = df['EURGBP 4. close'].shift(-1)
    else:
        df = pd.read_csv('./data/csv/dataset_long.csv', encoding='utf-8', index_col=0)
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
        idx = list(pd.date_range(df.index.min(), df.index.max()))
        df = df.reindex(idx, method='ffill')
        df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
        labels = df['EURGBP 4. close'].shift(-1)

    return df[pd.notnull(labels)], labels[pd.notnull(labels)]


def fetch_crypto_rate(from_currency, to_currency, start, end, freq):
    """
    Given currencies and start/end dates, as well as frequency, gets exchange rates from Poloniex API.
    """

    base_url = "https://poloniex.com/public?command=returnChartData" + "&currencyPair=" + from_currency + "_" + to_currency

    start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    tmp1 = start
    tmp2 = start + timedelta(weeks=12)
    while tmp2 <= end:
        x1, x2 = datetime.timestamp(tmp1), datetime.timestamp(tmp2)
        main_url = base_url + "&start=" + str(x1) + "&end=" + str(x2) + "&period=" + str(freq * 60)
        print('Fetching:', main_url)
        req_ob = requests.get(main_url)
        result = req_ob.json()

        d1, d2 = tmp1.strftime("%Y%m%d"), tmp2.strftime("%Y%m%d")
        with open('data/raw/crypto/' + d1 + '-' + d2 + '-' + from_currency + to_currency + str(freq) + '.json', 'w') as outfile:
            json.dump(result, outfile)

        tmp1, tmp2 = tmp1 + timedelta(weeks=12), tmp2 + timedelta(weeks=12)
        if tmp1 < end < tmp2:
            tmp2 = end


def fetch_exchange_rate(from_currency, to_currency, freq, api_key):
    """
    Given a currency pair, gets all possible historic data from Alphavantage.
    """

    base_url = r"https://www.alphavantage.co/query?function=FX_INTRADAY"
    main_url = base_url + "&from_symbol=" + from_currency + "&to_symbol=" + to_currency
    main_url = main_url + "&interval=" + str(freq) + "min&outputsize=full" + "&apikey=" + api_key
    print('Fetching:', main_url)
    req_ob = requests.get(main_url)
    result = req_ob.json()

    with open('data/raw/' + str(date.today()) + '-' + from_currency + to_currency + str(freq) + '.json',
              'w') as outfile:
        json.dump(result, outfile)


def structure_crypto(from_curr, to_curr, freq):
    """
    Once cryptocurrencies rate json file are obtained, structures it into standard csv.
    """

    df = pd.DataFrame()
    for f in glob.glob('./data/raw/crypto/*-*-' + from_curr + to_curr + str(freq) + '.json'):
        with open(f) as json_file:
            data = json.load(json_file)
            df1 = pd.DataFrame.from_dict(data).set_index('date')
            df1.index = pd.to_datetime(df1.index, unit='s')
            df = df.combine_first(df1)
    df.to_csv('./data/csv/dataset_crypto.csv', encoding='utf-8')


def structure_currencies(from_curr, to_curr, freq):
    """
    Once currencies exchange rate json file are obtained, structures it into standard csv.
    """

    dfA = pd.DataFrame()
    for f in glob.glob('./data/raw/*-*-' + from_curr + to_curr + str(freq) + '.json'):
        with open(f) as json_file:
            data = json.load(json_file)
            df1 = pd.DataFrame.from_dict(data["Time Series FX (" + str(freq) + "min)"], orient='index')
            dfA = dfA.combine_first(df1)
    dfA.columns = [(from_curr + to_curr + ' ' + x) for x in dfA.columns]

    dfB = pd.DataFrame()
    for f in glob.glob('./data/raw/*-*-' + to_curr + from_curr + str(freq) + '.json'):
        with open(f) as json_file:
            data = json.load(json_file)
            df1 = pd.DataFrame.from_dict(data["Time Series FX (" + str(freq) + "min)"], orient='index')
            dfB = dfB.combine_first(df1)
    dfB.columns = [(to_curr + from_curr + ' ' + x) for x in dfB.columns]

    if not dfA.index.equals(dfB.index):
        print('Warning: index not aligned btw exchange rate and reverse exchange rate.')
    df = pd.concat([dfA, dfB], axis=1, sort=True).dropna()

    df.to_csv('./data/csv/dataset.csv', encoding='utf-8')


def compute_metrics(y_true, y_pred):
    """
    Given true labels and predictions, outputs a set of performance metrics for regression task.
    """

    are = np.abs(y_pred - y_true) / np.abs(y_true)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    me = max_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {'max_abs_rel_error':    are.max(),
            'mean_abs_rel_error':   are.mean(),
            'r2':                   r2,
            'explained_var':        evs,
            'max_error':            me,
            'mean_squared_error':   mse,
            'mean_absolute_error':  mae
            }
