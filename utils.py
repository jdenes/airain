import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import font_manager
from datetime import datetime, timedelta
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score, classification_report

cmap = ['#f77189', '#e68332', '#bb9832', '#97a431', '#50b131', '#34af84', '#36ada4', '#38aabf', '#3ba3ec', '#a48cf4',
        '#e866f4', '#f668c2']

encode = {'Buy': 2, 'Outperform': 2, 'Overweight': 2, 'Sector Outperform': 2,
          'Market Perform': 1, 'Sector Perform': 1, 'Neutral': 1, 'Hold': 1, 'Equal-Weight': 1,
          'Underperform': 0, 'Sell': 0}


def load_data(filename, target_col, lag=0, tradefreq=1, datafreq=1, keep_last=False, enrich=True):
    """
    Given a data source, loads appropriate csv file.
    """

    # number, asset = filename
    filename = './data/dataset_eurusd_train_{}.csv'.format(str(datafreq))
    # filename = './data/finance/' + asset + '.csv'
    # filename = './data/jena_climate_2009_2016.csv'

    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    df.index = df.index.rename('date')
    df = df.loc[~df.index.duplicated(keep='last')]

    askcol, bidcol = 'ask' + str(target_col), 'bid' + str(target_col)
    # askcol, bidcol = 'T (degC)', 'p (mbar)'
    # askcoln bidcol = 'Close', 'Adj Close'

    # df = df['2010-01-01':]

    # ticker = yf.Ticker(asset)
    # reco = ticker.recommendations['To Grade'].apply(lambda x: encode[x]).rename('Analysis')
    # reco.index = reco.index.rename('date').strftime('%Y-%m-%d')
    # reco = reco.groupby('date').sum()
    # df = pd.concat([df, reco], join='outer', axis=1).sort_index()
    # df['Analysis'] = df['Analysis'].ffill().fillna(0)
    # df = df[(df.Analysis != 0).idxmax():]
    
    df['labels'] = (((df[askcol].shift(-tradefreq) - df[askcol]) / df[askcol]) > 0).astype(int)

    time_index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)
    df['year'] = time_index.year
    df['month'] = time_index.month - 1
    df['day'] = time_index.day - 1
    df['wday'] = time_index.weekday - 1
    df['hour'] = time_index.hour
    df['minute'] = time_index.minute

    if enrich:
        for col in df:
            if 'open' in col or 'close' in col:
                df[col + '_delta_1'] = df[col].diff(1)
                df[col + '_delta_7'] = df[col].diff(7)

    df[['mwa_25_ask', 'wma_25_bid']] = df[[askcol, bidcol]].ewm(alpha=0.25).mean()
    df[['mwa_75_ask', 'wma_75_bid']] = df[[askcol, bidcol]].ewm(alpha=0.75).mean()
    
    for lag in [1, 7, 14, 30, 60]:
        lag_col = 'lag_' + str(lag)
        df[lag_col + '_ask'] = df[askcol].shift(lag)
        df[lag_col + '_bid'] = df[bidcol].shift(lag)
        df[lag_col + '_labels'] = df['labels'].shift(lag)
        for win in [7, 28] :
            col = 'sma_{}_{}'.format(str(lag), str(win))
            df[col + '_ask'] = df[lag_col + '_ask'].transform(lambda x : x.rolling(win).mean())
            df[col + '_bid'] = df[lag_col + '_bid'].transform(lambda x : x.rolling(win).mean())
            df[col + '_labels'] = df[lag_col + '_labels'].transform(lambda x : x.rolling(win).mean())
            
    # df['asset'] = number
    # df['asset_mean'] = df['labels'].mean()
    # df['asset_std'] = df['labels'].std()
    
    for period in ['year', 'month', 'day', 'wday', 'hour', 'minute']:
        df[period + '_mean'] = df.groupby(period)['labels'].transform('mean')
        df[period + '_std'] = df.groupby(period)['labels'].transform('std')
        df[period + '_ask_mean'] = df.groupby(period)[askcol].transform('mean')
        df[period + '_bid_mean'] = df.groupby(period)[askcol].transform('mean')

    labels = df['labels']
    df = df.drop(['labels'], axis=1)

    # print(labels.value_counts(normalize=True))
    if keep_last:
        index = pd.notnull(df).all(1)
    else:
        index = pd.notnull(df).all(1) & pd.notnull(labels)

    return df[index], labels[index]


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
        df.index = pd.to_datetime(df.index, unit='s')
        df.to_csv(filename, encoding='utf-8')

    else:
        count = 0
        df = pd.DataFrame()
        step = (1 + 0.5 * int(freq != 1)) * freq
        start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        tmp1 = start
        if end - start < timedelta(weeks=step):
            tmp2 = end
        else:
            tmp2 = start + timedelta(weeks=step)
        while tmp2 <= end:
            df = con.get_candles(curr, period='m' + str(freq), start=tmp1, stop=tmp2)
            df.index = pd.to_datetime(df.index, unit='s')
            if count == 0:
                df.to_csv(filename, encoding='utf-8')
            else:
                df.to_csv(filename, encoding='utf-8', mode='a', header=False)                
            tmp1, tmp2 = tmp1 + timedelta(weeks=step), tmp2 + timedelta(weeks=step)
            count += 1
            if tmp1 < end < tmp2:
                tmp2 = end

    


def change_accuracy(y_true, y_pred):
    pred_shift = np.array(y_pred[1:] > y_true[:-1]).astype(int)
    true_shift = np.array(y_true[1:] > y_true[:-1]).astype(int)
    return (pred_shift == true_shift).mean()


def compute_metrics(y_true, y_pred):
    """
    Given true labels and predictions, outputs a set of performance metrics for regression task.
    """

    i = y_true != 0
    are = np.abs(y_pred[i] - y_true[i]) / np.abs(y_true[i])
    are = are[are < 1e8]
    r2 = r2_score(y_true, y_pred)
    me = max_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ca = change_accuracy(y_true, y_pred)

    return {'max_abs_rel_error': are.max(),
            'mean_abs_rel_error': are.mean(),
            'mean_absolute_error': mae,
            'max_error': me,
            'mean_squared_error': mse,
            'r2': r2,
            'change_accuracy': ca
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
    return 2 * (data - data_min) / (data_max - data_min) - 1


def unnormalize_data(data, data_max, data_min):
    """
    Un-normalizes data using min-max normalization.
    """
    return (data + 1) * (data_max - data_min) / 2 + data_min


def nice_plot(ind, curves_list, names_list, title):

    font_manager._rebuild()
    plt.rcParams['font.family'] = 'Lato'
    plt.rcParams['font.sans-serif'] = 'Lato'
    plt.rcParams['font.weight'] = 500

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, x in enumerate(curves_list):
        pd.Series(index=ind, data=list(x)).plot(linewidth=2, color=cmap[i], ax=ax, label=names_list[i])
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


def create_labels(df, col_name, window_size=11):
    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 0, HOLD => 2
    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy
    returns : numpy array with integer codes for labels with
              size = total-(window_size)+1
    """

    from tqdm.auto import tqdm
    labels = np.zeros(len(df))
    labels[:] = np.nan
    print("Computing labels...")
    pbar = tqdm(total=len(df))

    for i in range(len(df) - window_size):

        window_begin = i
        window_end = window_begin + window_size + 1
        window_middle = (window_begin + window_end) / 2

        price = df.iloc[window_begin:window_end][col_name]
        max_index = price.argmax() + window_begin
        min_index = price.argmin() + window_begin

        if max_index == window_middle:
            labels[i] = 2
        elif min_index == window_middle:
            labels[i] = 1
        else:
            labels[i] = 0
        pbar.update(1)

    pbar.close()
    return labels
