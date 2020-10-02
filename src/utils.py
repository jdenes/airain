import os
import logging
import requests
import warnings
import numpy as np
import pandas as pd
import joblib as jl

import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import font_manager
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)

CMAP = ['#f77189', '#e68332', '#bb9832', '#97a431', '#50b131', '#34af84', '#36ada4', '#38aabf', '#3ba3ec', '#a48cf4',
        '#e866f4', '#f668c2']

ENCODE = {'Buy': 2, 'Outperform': 2, 'Overweight': 2, 'Sector Outperform': 2,
          'Market Perform': 1, 'Sector Perform': 1, 'Neutral': 1, 'Hold': 1, 'Equal-Weight': 1,
          'Underperform': 0, 'Sell': 0}

KEYWORDS = {'AAPL': ['aap', 'apple', 'phone', 'mac', 'microsoft'],
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
            'NKE': ['nke', 'nike', 'sport', 'wear'],
            'CAT': ['cat', 'caterpillar', 'construction'],
            'V': ['visa', 'bank', 'card', 'pay'],
            'JPM': ['jpm', 'morgan', 'chase', 'bank'],
            'AXP': ['axp', 'american', 'express', 'bank', 'card', 'pay'],
            'GS': ['gs', 'goldman', 'sachs', 'bank'],
            'UNH': ['unh', 'united', 'health', 'insurance'],
            'TRV': ['trv', 'travel', 'insurance'],
            # 'UTX': ['utx', 'united', 'tech'],
            }


def load_data(folder, tradefreq=1, datafreq=1, start_from=None, update_embed=False):
    """
    Loads, data-engineers and concatenates each assets' data into a large machine-usable dataframe.

    :param str folder: a folder where to find the dataset.
    :param int tradefreq: at which tick frequency you will trade.
    :param int datafreq: at which frequency you want each tick, in minutes.
    :param None|str start_from: a starting date for truncation, in format '%Y-%m-%d'. If None, no truncation is made.
    :param bool update_embed: whether new news have been obtained and embeddings should be re-computed.
    :return: a rich dataframe.
    :rtype: pd.DataFrame
    """

    t1 = '2020-04-01'
    res = pd.DataFrame()

    for number, asset in enumerate(KEYWORDS.keys()):

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
            embed = load_news(asset, KEYWORDS[asset])
            embed.to_csv(path, encoding='utf-8')
        else:
            embed = pd.read_csv(path, encoding='utf-8', index_col=0)
            if update_embed:
                last_embed = embed.index.max()
                embed = load_news(asset, KEYWORDS[asset], last_day=last_embed)
                write_data(path, embed)
                embed = pd.read_csv(path, encoding='utf-8', index_col=0)

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

        shifted_askcol, shifted_bidcol = df[askcol].shift(-tradefreq), df[bidcol].shift(-tradefreq)
        df['labels'] = ((shifted_askcol - shifted_bidcol) > 0).astype(int)
        # df['labels'] = pd.concat([shifted_bidcol, shifted_askcol], axis=1).apply(compute_label, axis=1)

        time_index = pd.to_datetime(df.index.to_list(), format='%Y-%m-%d', utc=True)  # %H:%M:%S', utc=True)
        df['year'] = time_index.year
        df['month'] = time_index.month
        df['quarter'] = time_index.quarter
        df['day'] = time_index.day
        df['week'] = time_index.week
        df['mweek'] = time_index.map(week_of_month)
        df['wday'] = time_index.weekday
        df['dayofyear'] = time_index.dayofyear
        # df['hour'] = time_index.hour
        # df['minute'] = time_index.minute

        # i = df.index.get_loc(t1)

        for col in ['open', 'close']:
            df[col + '_delta_1'] = df[col].diff(1)
            df[col + '_delta_7'] = df[col].diff(7)

        for lag in [1, 7, 14, 30, 60]:
            lag_col = 'lag_' + str(lag)
            df[lag_col + '_ask'] = df[askcol].shift(lag)
            df[lag_col + '_bid'] = df[bidcol].shift(lag)
            df[lag_col + '_labels'] = df['labels'].shift(lag)
            for win in [7, 14, 30, 60]:
                col = f'sma_{str(lag)}_{str(win)}'
                df[col + '_ask'] = df[lag_col + '_ask'].transform(lambda x: x.rolling(win).mean())
                df[col + '_bid'] = df[lag_col + '_bid'].transform(lambda x: x.rolling(win).mean())
                df[col + '_labels'] = df[lag_col + '_labels'].transform(lambda x: x.rolling(win).mean())

        df['asset'] = number
        df['asset_mean'] = df.groupby('asset')['labels'].transform(lambda x: x[x.index < t1].mean())
        df['asset_std'] = df.groupby('asset')['labels'].transform(lambda x: x[x.index < t1].std())

        for period in ['year', 'quarter', 'week', 'month', 'wday']:  # 'hour', 'minute', 'dayofyear', 'day'
            for col in ['labels', 'volume', askcol, bidcol]:
                df[period + '_mean_' + col] = df.groupby(period)[col].transform(lambda x: x[x.index < t1].mean())
                df[period + '_std_' + col] = df.groupby(period)[col].transform(lambda x: x[x.index < t1].std())
                # if col != 'labels':
                #     df[f'{period}_adj_{col}'] = df[f'{period}_mean_{col}'] / df[f'{period}_std_{col}']
                #     df[f'{period}_diff_{col}'] = df[col] - df[f'{period}_mean_{col}']

        # essayer Kalman Filter
        res = pd.concat([res, df], axis=0)

    # Computing overall aggregate features
    res = res.rename_axis('date').sort_values(['date', 'asset'])

    for period in ['year', 'month', 'day', 'wday']:
        for col in ['labels', 'volume']:
            res[f'ov_{period}_mean_{col}'] = res.groupby(period)[col].transform(lambda x: x[x.index < t1].mean())
            res[f'ov_{period}_std_{col}'] = res.groupby(period)[col].transform(lambda x: x[x.index < t1].std())

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


def load_news(asset, keywords=None, use_weekends=True, last_day=None):
    """
    Loads a Sentence-BERT embedded, daily-aggregated version of a company's news.

    :param str asset: company's symbol (e.g. AAPL, MSFT)
    :param list[str] keywords: a list of keywords on which to filter the news to ensure relevance.
    :param bool use_weekends: whether news published during weekends should be used.
    :param None|str last_day: the last date before truncation, in format '%Y-%m-%d'. If None, no truncation is made.
    :return: A dataframe with each row corresponding to an embedded, daily-aggregated company's news.
    :rtype: pd.DataFrame
    """

    filename = '../data/intrinio/' + asset.lower() + '_news.csv'
    df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
    df.drop(['id', 'url', 'publication_date'], axis=1, inplace=True)

    if last_day is not None:
        df = df[last_day:]
    if keywords is not None:
        pattern = '(?i)' + '|'.join(keywords)
        df = df[df['title'].str.contains(pattern) | df['summary'].str.contains(pattern)]

    if use_weekends:
        df.index = [x if datetime.strptime(x, '%Y-%m-%d').weekday() < 5 else previous_day(x) for x in df.index]

    res = df[['title', 'summary']].groupby(df.index).apply(sum)

    mod = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    # tmp = mod.encode(res['title'], show_progress_bar=True)
    # embed_title = pd.DataFrame(tmp, index=res.index)
    # embed_title.columns = ['news_title_{}'.format(i) for i in embed_title]

    tmp = mod.encode(res['summary'], show_progress_bar=(last_day is None))
    embed_sum = pd.DataFrame(tmp, index=res.index)
    embed_sum.columns = ['news_sum_{}'.format(i) for i in embed_sum]

    # embed = pd.concat([embed_title, embed_sum], axis=1).sort_index()

    return embed_sum.sort_index()


def precompute_embeddings(folder):
    """
    Pre-computes and saves Sentence-BERT embeddings of news.

    :param str folder: the folder where to find the news to embed.
    :rtype: None
    """

    for number, asset in enumerate(KEYWORDS.keys()):
        path = folder + '{}_news_embed.csv'.format(asset.lower())
        if not os.path.exists(path):
            embed = load_news(asset, KEYWORDS[asset])
            embed.to_csv(path, encoding='utf-8')
        else:
            embed = pd.read_csv(path, encoding='utf-8', index_col=0)
            last_embed = embed.index.max()
            embed = load_news(asset, KEYWORDS[asset], last_day=last_embed)
            write_data(path, embed)


def train_sbert_pca(dim=100):
    """
    Trains and saves a PCA to reduce Sentence-BERT embeddings size.

    :param int dim: size of the reduced embedding.
    :return: a trained PCA model.
    :rtype: sklearn.decomposition.PCA
    """

    df = pd.DataFrame()
    for asset in KEYWORDS.keys():
        path = '../data/intrinio/{}_news_embed.csv'.format(asset.lower())
        embed = pd.read_csv(path, encoding='utf-8', index_col=0)
        df = pd.concat([df, embed], axis=0, ignore_index=True)
    pca = PCA(n_components=dim)
    pca.fit(df)
    jl.dump(pca, '../data/pca_sbert_{}.joblib'.format(dim))
    return pca


def fetch_intrinio_news(filename, api_key, company, update=False):
    """
    Fetches and saves Intrinio API to get assets' historical news.

    :param str filename: name of the file where to store the obtained data.
    :param str api_key: Intrinio personal API key.
    :param str company: company's symbol (e.g. AAPL, MSFT)
    :param bool update: whether you only want to update the data (gets only last 100 entries) or get all the history.
    :rtype: None
    """

    base_url = 'https://api-v2.intrinio.com/companies/{}/news?page_size=100&api_key={}'.format(company, api_key)
    url, next_page, data = base_url, 0, None
    while next_page is not None:
        while True:
            try:
                data = requests.get(url).json()
            except:
                continue
            break
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
        df = df.drop_duplicates(keep='last')
        df.to_csv(filename, encoding='utf-8')


def fetch_intrinio_prices(filename, api_key, company, update=False):
    """
    Fetches and saves Intrinio API to get assets' historical prices.

    :param str filename: name of the file where to store the obtained data.
    :param str api_key: Intrinio personal API key.
    :param str company: company's symbol (e.g. AAPL, MSFT)
    :param bool update: whether you only want to update the data (gets only last 100 entries) or get all the history.
    :rtype: None
    """

    base_url = 'https://api-v2.intrinio.com/securities/{}/prices?page_size=100&api_key={}'.format(company, api_key)
    url, next_page, data = base_url, 0, None
    while next_page is not None:
        while True:
            try:
                data = requests.get(url).json()
            except:
                continue
            break
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


def normalize_data(data, data_max, data_min):
    """
    Normalizes data using min-max normalization.

    :param int|float|pd.Series data: data array or scalar to un-normalize.
    :param int|float data_max: max value to use in min-max normalization.
    :param int|float data_min: data_max: min value to use in min-max normalization.
    :return: un-normalized data.
    :rtype: int|float|pd.Series
    """

    return 2 * (data - data_min) / (data_max - data_min) - 1


def unnormalize_data(data, data_max, data_min):
    """
    Un-normalizes data using min-max normalization.

    :param int|float|pd.Series data: data array or scalar to un-normalize.
    :param int|float data_max: max value used in min-max normalization.
    :param int|float data_min: data_max: min value used in min-max normalization.
    :return: un-normalized data.
    :rtype: int|float|pd.Series
    """

    return (data + 1) * (data_max - data_min) / 2 + data_min


def nice_plot(ind, curves_list, names_list, title):
    """
    Provides nice plot for profit backtest curves.

    :param list[str] ind: list of date for each backtest step, in format '%Y-%m-%d'.
    :param list[list[float]] curves_list: a list of lists to plot.
    :param list[str] names_list: list of names for each curve.
    :param str title: name of the plot.
    :rtype: None
    """

    font_manager._rebuild()
    plt.rcParams['font.family'] = 'Lato'
    plt.rcParams['font.sans-serif'] = 'Lato'
    plt.rcParams['font.weight'] = 500
    from datetime import datetime
    ind = [datetime.strptime(x, '%Y-%m-%d') for x in ind]
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
        pd.Series(index=ind, data=list(x)).plot(linewidth=2, color=CMAP[i], ax=ax, label=names_list[i], style='.-')
    ax.tick_params(labelsize=12)
    # formatter = dates.DateFormatter('%d/%m/%Y')
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


def week_of_month(dt):
    """
    Get which week of a month a day belongs to.

    :param datetime dt: the day you want to know which week of month it belongs to.
    :return: week of the month number.
    :rtype: int
    """

    dt = pd.to_datetime(dt)
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom/7))


def next_day(date):
    """
    Returns next banking day.

    :param str date: the day you want to know which is the next banking day, in format '%Y-%m-%d'.
    :return: previous banking day in format '%Y-%m-%d'.
    :rtype: str
    """

    date = datetime.strptime(date, '%Y-%m-%d')
    if date.weekday() == 4:
        res = date + timedelta(days=3)
    elif date.weekday() == 5:
        res = date + timedelta(days=2)
    else:
        res = date + timedelta(days=1)
    return res.strftime('%Y-%m-%d')


def previous_day(date):
    """
    Returns previous banking day.

    :param str date: the day you want to know which is the previous banking day, in format '%Y-%m-%d'.
    :return: previous banking day in format '%Y-%m-%d'.
    :rtype: str
    """

    date = datetime.strptime(date, '%Y-%m-%d')
    if date.weekday() == 0:
        res = date - timedelta(days=3)
    elif date.weekday() == 6:
        res = date - timedelta(days=2)
    else:
        res = date - timedelta(days=1)
    return res.strftime('%Y-%m-%d')


def write_data(path, new_row, same_ids=False):
    """
    Writes data as .csv by creating a file or adding to an existing one if relevant.

    :param str path: path of the file where to write the data.
    :param pd.DataFrame new_row: dataframe to write.
    :param bool same_ids: whether the data contains duplicated indexes.
    :rtype: None
    """

    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8', index_col=0)
        df = pd.concat([df, new_row], axis=0)
        if same_ids:
            df = df.drop_duplicates(keep='last')
        else:
            df = df.loc[~df.index.duplicated(keep='last')]
        df.sort_index().to_csv(path, encoding='utf-8')
    else:
        new_row.to_csv(path, encoding='utf-8')


def benchmark_metrics(initial_gamble, balance_hist, orders_hist):
    """
    Computes financial metrics for a single asset.

    :param int initial_gamble: initial amount invested in each asset.
    :param list[int|float] balance_hist: list of asset's balance history.
    :param list[int] orders_hist: list of binary orders at each time step.
    :return: dictionary with various metrics.
    :rtype: dict
    """

    # Final profit
    profit = round(balance_hist[-1] - initial_gamble, 2)

    # Profits and returns
    full_bal_hist = [initial_gamble] + balance_hist
    profits_hist = [round(full_bal_hist[i] - full_bal_hist[i-1], 2) for i in range(1, len(full_bal_hist))]
    returns_hist = [round(100 * profit / initial_gamble, 2) for profit in profits_hist]

    # Basic counts: nb of sell and buy days, and correctness
    buy_count, sell_count, correct_buy, correct_sell = 0, 0, 0, 0
    for i in range(len(balance_hist)):
        is_positive = int(profits_hist[i] >= 0)
        if orders_hist[i] == 1:
            buy_count += 1
            correct_buy += is_positive
        elif orders_hist[i] == 0:
            sell_count += 1
            correct_sell += is_positive
    buy_acc = round(100 * correct_buy / buy_count, 2) if buy_count != 0 else 'NA'
    sell_acc = round(100 * correct_sell / sell_count, 2) if sell_count != 0 else 'NA'

    # Overall daily scores
    pos_pls, neg_pls = [i for i in profits_hist if i >= 0], [i for i in profits_hist if i < 0]
    pos_days = round(100 * sum([i >= 0 for i in profits_hist]) / len(profits_hist), 2)
    mean_returns = round(sum(returns_hist) / len(returns_hist), 2)
    mean_win = round(sum(pos_pls) / len(pos_pls), 2) if len(pos_pls) != 0 else 'NA'
    mean_loss = round(sum(neg_pls) / len(neg_pls), 2) if len(neg_pls) != 0 else 'NA'
    max_win, max_loss = max(profits_hist), min(profits_hist)

    return {'profit': profit,
            'profits_history': profits_hist,
            'positive_days': pos_days,
            'mean_returns': mean_returns,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_accuracy': buy_acc,
            'sell_accuracy': sell_acc,
            'mean_wins': mean_win,
            'mean_loss': mean_loss,
            'max_loss': max_loss,
            'max_win': max_win
            }


def benchmark_portfolio_metric(initial_gamble, assets_balance_hist):
    """
    Computes financial metrics for a portfolio.

    :param int initial_gamble: initial amount invested in each asset.
    :param list[list[int|float]] assets_balance_hist: list of each asset's balance history.
    :return: dictionary with various metrics.
    :rtype: dict
    """

    assets_profits_hist, assets_returns = [], []
    for balance_hist in assets_balance_hist:
        full_bal_hist = [initial_gamble] + balance_hist
        profits_hist = [round(full_bal_hist[i] - full_bal_hist[i - 1], 2) for i in range(1, len(full_bal_hist))]
        returns_hist = [round(100 * profit / initial_gamble, 2) for profit in profits_hist]
        mean_returns = round(sum(returns_hist) / len(returns_hist), 2)
        assets_profits_hist.append(profits_hist), assets_returns.append(mean_returns)
    portfolio_pls_hist = [sum([pls[i] for pls in assets_profits_hist]) for i in range(len(assets_profits_hist[0]))]
    portfolio_mean_profits = round(sum(portfolio_pls_hist) / len(portfolio_pls_hist), 2)
    portfolio_positive_days = round(100 * len([x for x in portfolio_pls_hist if x > 0]) / len(portfolio_pls_hist), 2)
    assets_profits = [balance[-1] - initial_gamble for balance in assets_balance_hist]
    count_profitable_assets = len([x for x in assets_profits if x > 0])
    assets_mean_profits = round(sum(assets_profits) / len(assets_profits), 2)
    assets_mean_returns = round(sum(assets_returns) / len(assets_returns), 3)

    return {'assets_returns': assets_returns,
            'assets_mean_profits': assets_mean_profits,
            'assets_mean_returns': assets_mean_returns,
            'portfolio_mean_profits': portfolio_mean_profits,
            'portfolio_positive_days': portfolio_positive_days,
            'count_profitable_assets': count_profitable_assets,
            }
