import os
import requests
import numpy as np
import pandas as pd
import joblib as jl

import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import font_manager
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.decomposition import PCA

cmap = ['#f77189', '#e68332', '#bb9832', '#97a431', '#50b131', '#34af84', '#36ada4', '#38aabf', '#3ba3ec', '#a48cf4',
        '#e866f4', '#f668c2']

encode = {'Buy': 2, 'Outperform': 2, 'Overweight': 2, 'Sector Outperform': 2,
          'Market Perform': 1, 'Sector Perform': 1, 'Neutral': 1, 'Hold': 1, 'Equal-Weight': 1,
          'Underperform': 0, 'Sell': 0}

keywords = {'AAPL': ['aap', 'apple', 'phone', 'mac', 'microsoft'],
            'XOM': ['xom', 'exxon', 'mobil', 'petrol', 'gas', 'energy'],
            'KO': ['ko', 'coca', 'cola', 'pepsi', 'soda'],
            'INTC': ['intc', 'intel', 'chip', 'cpu', 'computer'],
            'WMT': ['wmt', 'walmart', 'food'],
            'MSFT': ['msft', 'microsoft', 'gates', 'apple', 'computer'],
            'IBM': ['ibm', 'business', 'machine'],
            'CVX': ['cvx', 'chevron', 'petrol', 'gas', 'energy'],
            'JNJ': ['jnj', 'johnson', 'health', 'medi', 'pharma'],
            'PG': ['pg', 'procter', 'gamble', 'health', 'care'],
            'PFE': ['pfe', 'pfizer', 'health', 'medi', 'pharma'],
            'VZ': ['vz', 'verizon', 'comm'],
            'BA': ['ba', 'boeing', 'plane', 'air'],
            'MRK': ['mrk', 'merck', 'health', 'medi', 'pharma'],
            'CSCO': ['csco', 'cisco', 'system', 'techn'],
            'HD': ['hd', 'home', 'depot', 'construction'],
            'MCD': ['mcd', 'donald', 'food', 'burger'],
            'MMM': ['mmm', '3m'],
            'GE': ['ge', 'general', 'electric', 'tech', 'energy'],
            'UTX': ['utx', 'united', 'tech'],
            'NKE': ['nke', 'nike', 'sport', 'wear'],
            'CAT': ['cat', 'caterpillar', 'construction'],
            'V': ['visa', 'bank', 'card', 'pay'],
            'JPM': ['jpm', 'morgan', 'chase', 'bank'],
            'AXP': ['axp', 'american', 'express', 'bank', 'card', 'pay'],
            'GS': ['gs', 'goldman', 'sachs', 'bank'],
            'UNH': ['unh', 'united', 'health', 'insurance'],
            'TRV': ['trv', 'travel', 'insurance']
            }


def load_data(folder, tradefreq=1, datafreq=1, start_from=None, update_embed=False):
    """
    Given a data source, loads appropriate csv file.
    """

    t1 = '2020-01-02'
    res = pd.DataFrame()

    for number, asset in enumerate(keywords.keys()):

        file = folder + asset.lower() + '_prices.csv'
        # filename = '../data/jena_climate_2009_2016.csv'

        df = pd.read_csv(file, encoding='utf-8', index_col=0)
        df.index = df.index.rename('date')
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        df.drop(['intraperiod', 'frequency'], axis=1, inplace=True)
        df.drop([col for col in df if 'adj' in col], axis=1, inplace=True)

        # askcol, bidcol = 'ask' + str(target_col), 'bid' + str(target_col)
        # askcol, bidcol = 'T (degC)', 'p (mbar)'
        askcol, bidcol = 'close', 'open'

        path = folder + '{}_news_embed.csv'.format(asset.lower())
        if not os.path.exists(path):
            embed = load_news(asset, keywords[asset])
            embed.to_csv(path, encoding='utf-8')
        else:
            embed = pd.read_csv(path, encoding='utf-8', index_col=0)
            if update_embed:
                last_embed = embed.index.max()
                embed = load_news(asset, keywords[asset], last_day=last_embed)
                embed.to_csv(path, encoding='utf-8', mode='a', header=False)
                embed = pd.read_csv(path, encoding='utf-8', index_col=0)
                embed = embed.loc[~embed.index.duplicated(keep='last')]
                embed.to_csv(path, encoding='utf-8')

        dim = 75
        path = '../data/pca_sbert_{}.joblib'.format(dim)
        if not os.path.exists(path):
            pca = train_sbert_pca(dim=dim)
        else:
            pca = jl.load(path)
        embed = pd.DataFrame(pca.transform(embed), index=embed.index)
        embed.columns = ['news_sum_{}'.format(i) for i in embed]

        df = pd.concat([df, embed], axis=1).sort_index()
        df[[c for c in embed]] = df[[c for c in embed]].fillna(method='ffill')
        df = df.dropna()
        del embed

        df['labels'] = ((df[askcol].shift(-tradefreq) - df[bidcol].shift(-tradefreq)) > 0).astype(int)

        time_index = pd.to_datetime(df.index, format='%Y-%m-%d', utc=True)  # %H:%M:%S', utc=True)
        df['year'] = time_index.year
        df['month'] = time_index.month - 1
        df['quarter'] = df['month'] // 3
        df['day'] = time_index.day - 1
        df['wday'] = time_index.weekday - 1
        # df['hour'] = time_index.hour
        # df['minute'] = time_index.minute

        i = df.index.get_loc(t1)

        for col in ['open', 'close']:
            df[col + '_delta_1'] = df[col].diff(1)
            df[col + '_delta_7'] = df[col].diff(7)

        for lag in [1, 7, 14, 30, 60]:
            lag_col = 'lag_' + str(lag)
            df[lag_col + '_ask'] = df[askcol].shift(lag)
            df[lag_col + '_bid'] = df[bidcol].shift(lag)
            df[lag_col + '_labels'] = df['labels'].shift(lag)
            for win in [7, 14, 30, 60]:
                col = 'sma_{}_{}'.format(str(lag), str(win))
                df[col + '_ask'] = df[lag_col + '_ask'].transform(lambda x: x.rolling(win).mean())
                df[col + '_bid'] = df[lag_col + '_bid'].transform(lambda x: x.rolling(win).mean())
                df[col + '_labels'] = df[lag_col + '_labels'].transform(lambda x: x.rolling(win).mean())

        df['asset'] = number
        df['asset_mean'] = df.groupby('asset')['labels'].transform(lambda x: x.iloc[:i].mean())  # .transform('mean')
        df['asset_std'] = df.groupby('asset')['labels'].transform(lambda x: x.iloc[:i].std())  # .transform('std')

        for period in ['year', 'month', 'day', 'wday']:  # 'hour', 'minute'
            for col in ['labels', 'volume', askcol, bidcol]:
                df[period + '_mean_' + col] = df.groupby(period)[col].transform(
                    lambda x: x.iloc[:i].mean())  # .transform('mean')
                df[period + '_std_' + col] = df.groupby(period)[col].transform(
                    lambda x: x.iloc[:i].std())  # .transform('std')

        # essayer Kalman Filter
        res = pd.concat([res, df], axis=0)

    # Computing overall aggregate features
    res = res.rename_axis('date').sort_values(['date', 'asset'])
    i = res.index.get_loc(t1).stop
    for period in ['year', 'day', 'wday']:
        for col in ['labels', 'volume']:
            res['ov_{}_mean_{}'.format(period, col)] = res.groupby(period)[col].transform(lambda x: x.iloc[:i].mean())
            res['ov_{}_std_{}'.format(period, col)] = res.groupby(period)[col].transform(lambda x: x.iloc[:i].std())

    if start_from is not None:
        res = res[start_from:]
        # res = res[res['asset'] == subset[0]]

    labels = res['labels']
    res = res.drop(['labels'], axis=1)
    if update_embed:
        index = pd.notnull(res).all(1)
    else:
        index = pd.notnull(res).all(1) & pd.notnull(labels)

    return res[index], labels[index]


def load_news(asset, keywords=None, last_day=None):
    filename = '../data/intrinio/' + asset.lower() + '_news.csv'
    df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
    df.drop(['id', 'url', 'publication_date'], axis=1, inplace=True)

    if last_day is not None:
        df = df[last_day:]
    if keywords is not None:
        pattern = '(?i)' + '|'.join(keywords)
        df = df[df['title'].str.contains(pattern) | df['summary'].str.contains(pattern)]
    res = df[['title', 'summary']].groupby(df.index).apply(sum)

    mod = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    # tmp = mod.encode(res['title'], show_progress_bar=True)
    # embed_title = pd.DataFrame(tmp, index=res.index)
    # embed_title.columns = ['news_title_{}'.format(i) for i in embed_title]

    tmp = mod.encode(res['summary'], show_progress_bar=True)
    embed_sum = pd.DataFrame(tmp, index=res.index)
    embed_sum.columns = ['news_sum_{}'.format(i) for i in embed_sum]

    # embed = pd.concat([embed_title, embed_sum], axis=1).sort_index()

    return embed_sum.sort_index()


def train_sbert_pca(dim=100):
    df = pd.DataFrame()
    for asset in keywords.keys():
        path = '../data/intrinio/{}_news_embed.csv'.format(asset.lower())
        embed = pd.read_csv(path, encoding='utf-8', index_col=0)
        df = pd.concat([df, embed], axis=0, ignore_index=True)
    pca = PCA(n_components=dim)
    pca.fit(df)
    jl.dump(pca, '../data/pca_sbert_{}.joblib'.format(dim))
    return pca


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
    base_url += "&interval={}min&outputsize=full&apikey={}".format(str(freq), api_key)

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


def fetch_fxcm_data(filename, curr, freq, con, unit='m', start=None, end=None, n_last=None):
    """
    Given currencies and start/end dates, as well as frequency, gets exchange rates from FXCM API.
    """

    if n_last is not None:
        df = con.get_candles(curr, period=unit + str(freq), number=n_last)
        df.index = pd.to_datetime(df.index, unit='s')
        df.to_csv(filename, encoding='utf-8')

    else:
        count = 0
        step = (1 + 0.5 * int(freq != 1)) * freq
        start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        tmp1 = start
        if end - start < timedelta(weeks=step):
            tmp2 = end
        else:
            tmp2 = start + timedelta(weeks=step)
        while tmp2 <= end:
            df = con.get_candles(curr, period=unit + str(freq), start=tmp1, stop=tmp2)
            df.index = pd.to_datetime(df.index, unit='s')
            if count == 0:
                df.to_csv(filename, encoding='utf-8')
            else:
                df.to_csv(filename, encoding='utf-8', mode='a', header=False)
            tmp1, tmp2 = tmp1 + timedelta(weeks=step), tmp2 + timedelta(weeks=step)
            count += 1
            if tmp1 < end < tmp2:
                tmp2 = end


def fetch_intrinio_news(filename, api_key, company, update=False):
    base_url = 'https://api-v2.intrinio.com/companies/{}/news?page_size=100&api_key={}'.format(company, api_key)
    url = base_url
    next_page = 0
    while next_page is not None:
        data = requests.get(url).json()
        df = pd.DataFrame(data['news'])
        df = df[df['summary'].str.len() > 180]
        df['date'] = df['publication_date'].str[:10]
        df = df.set_index('date', drop=True)
        if next_page == 0 and not update:
            df.to_csv(filename, encoding='utf-8')
        else:
            df.to_csv(filename, encoding='utf-8', mode='a', header=False)
        next_page = data['next_page'] if not update else None
        url = base_url + '&next_page={}'.format(str(next_page))
    if update:
        df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
        df = df.drop_duplicates()
        df.to_csv(filename, encoding='utf-8')


def fetch_intrinio_prices(filename, api_key, company, update=False):
    base_url = 'https://api-v2.intrinio.com/securities/{}/prices?page_size=100&api_key={}'.format(company, api_key)
    url = base_url
    next_page = 0
    while next_page is not None:
        data = requests.get(url).json()
        df = pd.DataFrame(data['stock_prices']).set_index('date', drop=True)
        if next_page == 0 and not update:
            df.to_csv(filename, encoding='utf-8')
        else:
            df.to_csv(filename, encoding='utf-8', mode='a', header=False)
        next_page = data['next_page'] if not update else None
        url = base_url + '&next_page={}'.format(str(next_page))
    if update:
        df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        df.to_csv(filename, encoding='utf-8')


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
    from datetime import datetime
    ind = [datetime.strptime(x, '%Y-%m-%d') for x in ind]
    formatter = dates.DateFormatter('%d/%m/%Y')
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, x in enumerate(curves_list):
        # c_list = ['gray']
        # for j in range(len(x) - 1):
        #     start, stop = x[j], x[j + 1]
        #     color = ['green', 'red', 'orange'][0 if stop - start > 0 else 1 if stop - start < 0 else 2]
        #     c_list.append(color)
        #     ax.plot([ind[j], ind[j + 1]], [start, stop], linewidth=2, color=color)
        #     # ax.fill_between([ind[j], ind[j+1]], [start, stop], alpha=0.2, color=color)
        # for j in range(len(x)):
        #     ax.plot(ind[j], x[j], '.', color=c_list[j])
        pd.Series(index=ind, data=list(x)).plot(linewidth=2, color=cmap[i], ax=ax, label=names_list[i], style='.-')
    ax.tick_params(labelsize=12)
    # ax.xaxis.set_major_formatter(formatter)
    ax.spines['right'].set_edgecolor('lightgray')
    ax.spines['top'].set_edgecolor('lightgray')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.title(title, fontweight=500, fontsize=25, loc='left')
    plt.legend(loc='upper left', fontsize=15)
    plt.grid(alpha=0.3)
    # plt.savefig(path, bbox_inches='tight',format="png", dpi=300, transparent=True)
    plt.show()


def merge_finance_csv(folder='../data/finance', filename='../data/finance/global.csv'):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    res = []
    for i, f in enumerate(files):
        df = pd.read_csv(os.path.join(folder, f), encoding='utf-8', index_col=0)
        df['asset'] = f[:-4]
        res.append(df)
    res = pd.concat(res, axis=0, ignore_index=True)
    # df = df.set_index('Date')
    # df.index = pd.to_datetime(df.index, unit='s')
    res.to_csv(filename, encoding='utf-8')


def append_data(path, new_row):
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8', index_col=0)
        df = pd.concat([df, new_row], axis=0)
        df = df.loc[~df.index.duplicated(keep='last')]
        df.to_csv(path, encoding='utf-8')
    else:
        new_row.to_csv(path, encoding='utf-8')
