import os
import joblib
import pandas as pd

from datetime import datetime

from utils.basics import write_data, clean_string
from utils.dates import week_of_month, previous_day
from utils.constants import COMPANIES_KEYWORDS
from utils.logging import get_logger

logger = get_logger()


def load_data(folder, companies, t0, t1, start_from=None, keep_last=False):
    """
    Loads, data-engineers and concatenates each assets' data into a large machine-usable dataframe.

    :param str folder: a folder where to find the dataset.
    :param list[str] companies: list of companies to use (refer to utils.constants).
    :param str t0: starting date of the data, formatted YYYY-MM-DD.
    :param str t1: date before which aggregated features must be computed, formatted YYYY-MM-DD.
    :param None|str start_from: a starting date for truncation, formatted YYYY-MM-DD. If None, no truncation is made.
    :param bool keep_last: whether new news have been obtained and embeddings should be re-computed.
    :return: a rich dataframe.
    :rtype: pd.DataFrame
    """

    res = pd.DataFrame()
    askcol, bidcol = 'close', 'open'

    for number, asset in enumerate(companies):

        file = f'{folder}{asset.lower()}_prices.csv'

        # Chosen data
        df = pd.read_csv(file, encoding='utf-8', index_col=0)
        df.index = df.index.rename('date')
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df['ratio'] = (df['close'] / df['open']) - 1
        df_index = df.index

        # Really basic
        df['asset'] = number
        # df['labels'] = ((df[askcol].shift(-1) - df[bidcol].shift(-1)) > 0).astype(int)
        df['labels'] = df['close'].shift(-1) / df['close'].shift(0)  # relative change

        """ Add today results of Nikkei225 """
        jpn = pd.read_csv('../data/yahoo/^n225_prices.csv', encoding='utf-8', index_col=0)
        jpn.index = jpn.index.rename('date')
        jpn = jpn[['open', 'high', 'low', 'close']]
        jpn.columns = ['jpn_open', 'jpn_high', 'jpn_low', 'jpn_close']
        jpn['jpn_labels'] = ((jpn['jpn_close'] - jpn['jpn_open']) > 0).astype(int)
        jpn = jpn.loc[~jpn.index.duplicated(keep='last')].sort_index()
        jpn = jpn.shift(-1).drop(jpn.tail(1).index)
        df = pd.concat([df, jpn], axis=1)
        df[[c for c in jpn]] = df[[c for c in jpn]].fillna(method='bfill')
        df = df.loc[df_index].sort_index()

        """ Time features """
        try:
            time_index = pd.to_datetime(df.index.to_list(), format='%Y-%m-%d', utc=True)
        except ValueError:
            time_index = pd.to_datetime(df.index.to_list(), format='%Y-%m-%d %H:%M:%S', utc=True)
        # df['year'] = time_index.year
        df['month'] = time_index.month
        df['quarter'] = time_index.quarter
        df['day'] = time_index.day
        df['week'] = time_index.isocalendar()['week'].to_list()
        df['mweek'] = time_index.map(week_of_month)
        df['wday'] = time_index.weekday
        df['dayofyear'] = time_index.dayofyear

        """ Filter time data """
        df = df[df.index >= t0]
        df = df[df['wday'] < 5]

        """ Delta features """
        for col in ['open', 'close']:
            df[col + '_delta_1'] = df[col].diff(1)
            df[col + '_delta_7'] = df[col].diff(7)

        """ Lag and SMA features """
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

        """ Asset features """
        # df['asset_mean'] = df.groupby('asset')['labels'].transform(lambda x: x[x.index < t1].mean())
        # df['asset_std'] = df.groupby('asset')['labels'].transform(lambda x: x[x.index < t1].std())

        """ Aggregate time features """
        # for period in ['quarter', 'week', 'month', 'wday']:  # 'hour', 'minute', 'dayofyear', 'day', 'year'
        #     for col in ['labels', 'volume', askcol, bidcol]:
        #         df[period + '_mean_' + col] = df.groupby(period)[col].transform(lambda x: x[x.index < t1].mean())
        #         df[period + '_std_' + col] = df.groupby(period)[col].transform(lambda x: x[x.index < t1].std())

        """ News embeddings """
        # path = f'../data/intrinio/{asset.lower()}_news_embed.csv'
        # if not os.path.exists(path):
        #     embed = load_news(asset, folder, COMPANIES_KEYWORDS[asset])
        #     embed.to_csv(path, encoding='utf-8')
        # else:
        #     embed = pd.read_csv(path, encoding='utf-8', index_col=0)
        #     if keep_last:
        #         last_embed = embed.index.max()
        #         embed = load_news(asset, folder, COMPANIES_KEYWORDS[asset], last_day=last_embed)
        #         write_data(path, embed)
        #         embed = pd.read_csv(path, encoding='utf-8', index_col=0)
        # dim = 60
        # pca_path = f"../data/pca_sbert_{dim}.joblib"
        # if not os.path.exists(pca_path):
        #     pca = train_sbert_pca(t1=t1, dim=dim)
        # else:
        #     pca = joblib.load(pca_path)
        # embed = pd.DataFrame(pca.transform(embed), index=embed.index)
        # embed.columns = [f"news_sum_{i}" for i in embed]

        """ Sentiment inference """
        # path = f'{folder}{asset.lower()}_news_sentiment.csv'
        # if not os.path.exists(path):
        #     sentiment = sentiment_analysis(asset, folder, COMPANIES_KEYWORDS[asset])
        #     sentiment.to_csv(path, encoding='utf-8')
        # else:
        #     sentiment = pd.read_csv(path, encoding='utf-8', index_col=0)
        #     if keep_last:
        #         last_embed = sentiment.index.max()
        #         sentiment = sentiment_analysis(asset, folder, COMPANIES_KEYWORDS[asset], last_day=last_embed)
        #         write_data(path, sentiment)
        #         sentiment = pd.read_csv(path, encoding='utf-8', index_col=0)

        # df = pd.concat([df, embed], axis=1).sort_index()
        # df[[c for c in embed]] = df[[c for c in embed]].fillna(method='ffill')
        # # df = df[df[[c for c in df if c not in embed.columns and c != 'labels']].notnull().all(axis=1)]
        # del embed

        # import numpy as np
        # mat = df[[c for c in df if 'news' in c]].to_numpy()
        # df['news_entropy'] = -np.sum(mat * np.log(mat, where=(mat > 0)), axis=1)

        """ Hodrick-Prescott filter """
        # import statsmodels.api as sm
        # for count, i in enumerate(df.index):
        #     if count > 5:
        #         for col in ['volume', askcol, bidcol]:
        #             cycle, trend = sm.tsa.filters.hpfilter(df[df.index <= i][col], 1e6)
        #             df.loc[i, col + '_cycle'], df.loc[i, col + '_trend'] = cycle[-1], trend[-1]

        # df = df[df[[c for c in df if c != 'labels']].notnull().all(axis=1)]
        res = pd.concat([res, df], axis=0)

    # Computing overall aggregate features
    res = res.rename_axis('date').sort_values(['date', 'asset'])

    """ Overall aggregated time features """
    # for period in ['month', 'day', 'wday']:  # 'year'
    #     for col in ['labels', 'volume']:
    #         res[f'ov_{period}_mean_{col}'] = res.groupby(period)[col].transform(lambda x: x[x.index < t1].mean())
    #         res[f'ov_{period}_std_{col}'] = res.groupby(period)[col].transform(lambda x: x[x.index < t1].std())
    # for col in ['lag_1_labels', 'volume']:
    #     res[f'ov_mean_{col}'] = res.groupby(res.index)[col].transform(lambda x: x.mean())

    if start_from is not None:
        res = res[res.index >= start_from]

    labels = res['labels']
    res = res.drop(['labels'], axis=1)

    if keep_last:
        index = pd.notnull(res).all(axis=1)
    else:
        index = pd.notnull(res).all(axis=1) & pd.notnull(labels)

    return res[index], labels[index]


def load_news(asset, folder, keywords=None, use_weekends=False, last_day=None):
    """
    Loads a Sentence-BERT embedded, daily-aggregated version of a company's news.

    :param str asset: company's symbol (e.g. AAPL, MSFT)
    :param str folder: folder where data are stored.
    :param list[str] keywords: a list of keywords on which to filter the news to ensure relevance.
    :param bool use_weekends: whether news published during weekends should be used.
    :param None|str last_day: the last date before truncation, in format '%Y-%m-%d'. If None, no truncation is made.
    :return: A dataframe with each row corresponding to an embedded, daily-aggregated company's news.
    :rtype: pd.DataFrame
    """

    from sentence_transformers import SentenceTransformer
    news = preprocess_news(asset=asset, folder=folder, keywords=keywords, use_weekends=use_weekends, last_day=last_day)
    mod = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    tmp = mod.encode(news['summary'], batch_size=32, show_progress_bar=(last_day is None))
    embed_sum = pd.DataFrame(tmp, index=news.index)
    embed_sum.columns = [f'news_sum_{i}' for i in embed_sum]
    # embed = pd.concat([embed_title, embed_sum], axis=1).sort_index()
    return embed_sum.groupby(embed_sum.index).mean()


def sentiment_analysis(asset, folder, keywords=None, use_weekends=False, last_day=None):
    """
    Loads a sentiment-analyzed, daily-aggregated version of a company's news.

    :param str asset: company's symbol (e.g. AAPL, MSFT)
    :param str folder: folder where data are stored.
    :param list[str] keywords: a list of keywords on which to filter the news to ensure relevance.
    :param bool use_weekends: whether news published during weekends should be used.
    :param None|str last_day: the last date before truncation, in format '%Y-%m-%d'. If None, no truncation is made.
    :return: A dataframe with each row corresponding to a pos/neg sentiment analysis, daily-aggregated company's news.
    :rtype: pd.DataFrame
    """

    import flair
    news = preprocess_news(asset=asset, folder=folder, keywords=keywords, use_weekends=use_weekends, last_day=last_day)
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

    sentences = [flair.data.Sentence(s) for s in news['title']]
    flair_sentiment.predict(sentences, verbose=(last_day is None))
    title_sentiment = [(1 if s.labels[0].value == 'POSITIVE' else -1) * s.labels[0].score for s in sentences]

    sentences = [flair.data.Sentence(s) for s in news['summary']]
    flair_sentiment.predict(sentences, verbose=(last_day is None))
    summary_sentiment = [(1 if s.labels[0].value == 'POSITIVE' else -1) * s.labels[0].score for s in sentences]

    title_sentiment = pd.DataFrame(title_sentiment, index=news.index)
    summary_sentiment = pd.DataFrame(summary_sentiment, index=news.index)
    sentiment = pd.concat([title_sentiment, summary_sentiment], axis=1).sort_index()
    sentiment.columns = ['sentiment_title', 'sentiment_summary']
    return sentiment.groupby(sentiment.index).mean()


def preprocess_news(asset, folder, keywords=None, aggregate=False, use_weekends=False, last_day=None):
    """
    Preprocesses company's news (daily-aggregated) to be used in other functions.

    :param str asset: company's symbol (e.g. AAPL, MSFT)
    :param str folder: folder where data are stored.
    :param list[str] keywords: a list of keywords on which to filter the news to ensure relevance.
    :param bool aggregate: whether to daily-aggregate pieces of news.
    :param bool use_weekends: whether news published during weekends should be used.
    :param None|str last_day: the last date before truncation, in format '%Y-%m-%d'. If None, no truncation is made.
    :return: A dataframe with each row corresponding the daily-aggregated company's news.
    :rtype: pd.DataFrame
    """

    filename = folder + asset.lower() + '_news.csv'
    df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
    df = df[df['title'].notnull() & df['summary'].notnull()]
    # df.drop(['id', 'url', 'publication_date'], axis=1, inplace=True)
    df['summary'] = df['summary'].apply(clean_string)
    if last_day is not None:
        df = df[last_day:]
    if keywords is not None:
        pattern = '(?i)' + '|'.join(keywords)
        df = df[df['title'].str.contains(pattern) | df['summary'].str.contains(pattern)]
    if use_weekends:
        df.index = [x if datetime.strptime(x, '%Y-%m-%d').weekday() < 5 else previous_day(x) for x in df.index]
    if aggregate:
        news = df[['title', 'summary']].groupby(df.index).apply(sum)
    else:
        news = df[['title', 'summary']]
    return news


def precompute_embeddings(folder):
    """
    Pre-computes and saves Sentence-BERT embeddings of news.

    :param str folder: the folder where to find the news to embed.
    :rtype: None
    """

    for number, asset in enumerate(COMPANIES_KEYWORDS.keys()):
        path = folder + f'{asset.lower()}_news_embed.csv'
        if not os.path.exists(path):
            embed = load_news(asset, folder, COMPANIES_KEYWORDS[asset])
            embed.to_csv(path, encoding='utf-8')
        else:
            embed = pd.read_csv(path, encoding='utf-8', index_col=0)
            last_embed = embed.index.max()
            embed = load_news(asset, folder, COMPANIES_KEYWORDS[asset], last_day=last_embed)
            write_data(path, embed)


def train_sbert_pca(t1, dim=100):
    """
    Trains and saves a PCA to reduce Sentence-BERT embeddings size.

    :param str t1: date before which aggregated features must be computed, formatted YYYY-MM-DD.
    :param int dim: size of the reduced embedding.
    :return: a trained PCA model.
    :rtype: sklearn.decomposition.PCA
    """

    from sklearn.decomposition import PCA
    df = pd.DataFrame()
    for asset in COMPANIES_KEYWORDS.keys():
        path = f'../data/intrinio/{asset.lower()}_news_embed.csv'
        embed = pd.read_csv(path, encoding='utf-8', index_col=0)
        embed = embed[embed.index < t1]
        df = pd.concat([df, embed], axis=0, ignore_index=True)
    pca = PCA(n_components=dim)
    pca.fit(df)
    joblib.dump(pca, f'../data/pca_sbert_{dim}.joblib')
    return pca
