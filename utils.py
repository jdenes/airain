import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from datetime import datetime, timedelta
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

cmap = ['#f77189', '#e68332', '#bb9832', '#97a431', '#50b131', '#34af84', '#36ada4', '#38aabf', '#3ba3ec', '#a48cf4',
        '#e866f4', '#f668c2']


def load_data(filename, target_col, lag=0, tradefreq=1, datafreq=1, keep_last=False, enrich=True):
    """
    Given a data source, loads appropriate csv file.
    """

    filename = './data/' + filename + '_' + str(datafreq) + '.csv'
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    df.index = df.index.rename('date')
    df = df.loc[~df.index.duplicated(keep='last')]

    if enrich:
        for col in df:
            if 'open' in col or 'close' in col:
                df[col + 'delta'] = df[col].diff()

    ask_fut = df['askopen'].shift(-lag-tradefreq)
    bid_fut = df['bidopen'].shift(-lag-tradefreq)
    ask_now = df['askopen'].shift(-lag)
    bid_now = df['bidopen'].shift(-lag)

    labels = df['askopen'].copy()
    for i in ask_fut.index:
        if bid_fut[i] > ask_now[i]:
            labels[i] = 1
        elif ask_fut[i] < bid_now[i]:
            labels[i] = 2
        else:
            labels[i] = 0

    # labels = (df[target_col].shift(-lag-tradefreq) > df[target_col].shift(-lag)).astype(int)
    prices = pd.concat((df['askopen'].shift(-lag),
                        df['bidopen'].shift(-lag),
                        df['askopen'].rolling(5).mean(),
                        df['bidopen'].rolling(5).mean(),
                        df['askopen'].rolling(30).mean(),
                        df['bidopen'].rolling(30).mean(),
                        df['askopen'].ewm(alpha=0.25).mean(),
                        df['bidopen'].ewm(alpha=0.25).mean(),
                        df['askopen'].ewm(alpha=0.75).mean(),
                        df['bidopen'].ewm(alpha=0.75).mean()
                        ), axis=1)

    if keep_last:
        index = pd.notnull(df).all(1)
    else:
        index = pd.notnull(df).all(1) & pd.notnull(prices).all(1) & pd.notnull(labels)

    print(labels[index].value_counts(normalize=True))
    return df[index], labels[index], prices[index]


def fetch_crypto_rate(filename, from_currency, to_currency, start, end, freq):
    """
    Given currencies and start/end dates, as well as frequency, gets exchange rates from Poloniex API.
    """

    base_url = "https://poloniex.com/public?command=returnChartData"
    base_url += "&currencyPair=" + from_currency + "_" + to_currency
    data = []

    start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    tmp1 = start
    if end - start < timedelta(weeks=2 * int(freq)):
        tmp2 = end
    else:
        tmp2 = start + timedelta(weeks=2 * int(freq))
    while tmp2 <= end:
        x1, x2 = datetime.timestamp(tmp1), datetime.timestamp(tmp2)
        main_url = base_url + "&start=" + str(x1) + "&end=" + str(x2) + "&period=" + str(freq * 60)
        print('Fetching:', main_url)
        try:
            res = requests.get(main_url).json()
            if res[0]['date'] != 0:
                data += requests.get(main_url).json()
        except:
            raise ValueError('Unable to fetch data, please check connection and API availability.')

        tmp1, tmp2 = tmp1 + timedelta(weeks=2 * int(freq)), tmp2 + timedelta(weeks=2 * int(freq))
        if tmp1 < end < tmp2:
            tmp2 = end

    df = pd.DataFrame.from_dict(data).set_index('date')
    df.index = pd.to_datetime(df.index, unit='s')  # .tz_localize('UTC').tz_convert('Europe/Paris')
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


def fetch_fxcm_data(filename, curr, freq, con, start=None, end=None, n_last=None):
    """
    Given currencies and start/end dates, as well as frequency, gets exchange rates from FXCM API.
    """

    if n_last is not None:
        df = con.get_candles(curr, period='m' + str(freq), number=n_last)

    else:
        df = pd.DataFrame()
        step = (1 + 0.5 * int(freq != 1)) * freq
        start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        tmp1 = start
        if end - start < timedelta(weeks=step):
            tmp2 = end
        else:
            tmp2 = start + timedelta(weeks=step)
        while tmp2 <= end:
            data = con.get_candles(curr, period='m' + str(freq), start=tmp1, stop=tmp2)
            df = pd.concat([df, data]).drop_duplicates(keep='last')

            tmp1, tmp2 = tmp1 + timedelta(weeks=step), tmp2 + timedelta(weeks=step)
            if tmp1 < end < tmp2:
                tmp2 = end

    df.index = pd.to_datetime(df.index, unit='s')
    df.to_csv(filename, encoding='utf-8')


def compute_metrics(y_true, y_pred):
    """
    Given true labels and predictions, outputs a set of performance metrics for regression task.
    """

    are = np.abs(y_pred - y_true) / np.abs(y_true)
    are = are[are < 1e8]
    r2 = r2_score(y_true, y_pred)
    me = max_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pred_shift = np.array(y_pred[:-1] > y_true[1:]).astype(int)
    true_shift = np.array(y_true[:-1] > y_true[1:]).astype(int)

    return {'max_abs_rel_error': are.max(),
            'mean_abs_rel_error': are.mean(),
            'mean_absolute_error': mae,
            'max_error': me,
            'mean_squared_error': mse,
            'r2': r2,
            'change_accuracy': (pred_shift == true_shift).mean()
            }


def evaluate(portfolio, rate):
    """
    Given a portfolio in units of base and quote currencies, returns value in base currency.
    """
    return round(portfolio[0] + (portfolio[1] * rate), 5)


def normalize_data(data, data_max, data_min):
    """
    Normalizes data using min-max normalization.
    """
    return 1 * (data - data_min) / (data_max - data_min) - 0


def unnormalize_data(data, data_max, data_min):
    """
    Un-normalizes data using min-max normalization.
    """
    return (data + 0) * (data_max - data_min) / 1 + data_min


def nice_plot(ind, curves_list, names, title):

    font_manager._rebuild()
    plt.rcParams['font.family'] = 'Lato'
    plt.rcParams['font.sans-serif'] = 'Lato'
    plt.rcParams['font.weight'] = 500

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, x in enumerate(curves_list):
        pd.Series(index=ind, data=list(x)).plot(linewidth=2, color=cmap[i], ax=ax, label=names[i])
    ax.tick_params(labelsize=12)
    ax.set_xticklabels([item.get_text()[5:-3] for item in ax.get_xticklabels()])
    ax.spines['right'].set_edgecolor('lightgray')
    ax.spines['top'].set_edgecolor('lightgray')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.title(title, fontweight=500, fontsize=25, loc='left')
    plt.legend(loc='upper left', fontsize=15)
    plt.grid(alpha=0.3)
    # plt.savefig(path, bbox_inches='tight',format="png", dpi=300, transparent=True)
    plt.show()
