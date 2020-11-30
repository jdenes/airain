import os
import flair
import joblib
import pandas as pd

from datetime import datetime
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from utils.basics import write_data, clean_string
from utils.dates import week_of_month, previous_day
from utils.logging import get_logger

logger = get_logger()

T1 = '2020-04-01'

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

    res = pd.DataFrame()
    truc = []

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

        # path = folder + '{}_news_sentiment.csv'.format(asset.lower())
        # if not os.path.exists(path):
        #     sentiment = sentiment_analysis(asset, KEYWORDS[asset])
        #     sentiment.to_csv(path, encoding='utf-8')
        # else:
        #     sentiment = pd.read_csv(path, encoding='utf-8', index_col=0)
        #     if update_embed:
        #         last_embed = sentiment.index.max()
        #         sentiment = sentiment_analysis(asset, KEYWORDS[asset], last_day=last_embed)
        #         write_data(path, sentiment)
        #         sentiment = pd.read_csv(path, encoding='utf-8', index_col=0)

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

        dim = 60
        pca_path = f"../data/pca_sbert_{dim}.joblib"
        if not os.path.exists(pca_path):
            pca = train_sbert_pca(dim=dim)
        else:
            pca = joblib.load(pca_path)
        embed = pd.DataFrame(pca.transform(embed), index=embed.index)
        embed.columns = [f"news_sum_{i}" for i in embed]

        df = pd.concat([df, embed], axis=1).sort_index()
        # df[[c for c in sentiment]] = df[[c for c in sentiment]].fillna(method='ffill')
        df[[c for c in embed]] = df[[c for c in embed]].fillna(method='ffill')
        df = df.dropna()
        del embed

        shifted_askcol, shifted_bidcol = df[askcol].shift(-tradefreq), df[bidcol].shift(-tradefreq)
        df['labels'] = ((shifted_askcol - shifted_bidcol) > 0).astype(int)

        time_index = pd.to_datetime(df.index.to_list(), format='%Y-%m-%d', utc=True)  # %H:%M:%S', utc=True)
        df['year'] = time_index.year
        df['month'] = time_index.month
        df['quarter'] = time_index.quarter
        df['day'] = time_index.day
        df['week'] = time_index.week
        df['mweek'] = time_index.map(week_of_month)
        df['wday'] = time_index.weekday
        df['dayofyear'] = time_index.dayofyear

        # i = df.index.get_loc(T1)

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
        df['asset_mean'] = df.groupby('asset')['labels'].transform(lambda x: x[x.index < T1].mean())
        df['asset_std'] = df.groupby('asset')['labels'].transform(lambda x: x[x.index < T1].std())

        for period in ['year', 'quarter', 'week', 'month', 'wday']:  # 'hour', 'minute', 'dayofyear', 'day'
            for col in ['labels', 'volume', askcol, bidcol]:
                df[period + '_mean_' + col] = df.groupby(period)[col].transform(lambda x: x[x.index < T1].mean())
                df[period + '_std_' + col] = df.groupby(period)[col].transform(lambda x: x[x.index < T1].std())
                # if col != 'labels':
                #     df[f'{period}_adj_{col}'] = df[f'{period}_mean_{col}'] / df[f'{period}_std_{col}']
                #     df[f'{period}_diff_{col}'] = df[col] - df[f'{period}_mean_{col}']

        # Hodrick-Prescott filter
        # for count, i in enumerate(df.index):
        #     if count > 5:
        #         for col in ['volume', askcol, bidcol]:
        #             cycle, trend = sm.tsa.filters.hpfilter(df[df.index <= i][col], 1600)
        #             df.loc[i, col + '_cycle'], df.loc[i, col + '_trend'] = cycle[-1], trend[-1]

        res = pd.concat([res, df], axis=0)

    # Computing overall aggregate features
    res = res.rename_axis('date').sort_values(['date', 'asset'])

    for period in ['year', 'month', 'day', 'wday']:
        for col in ['labels', 'volume']:
            res[f'ov_{period}_mean_{col}'] = res.groupby(period)[col].transform(lambda x: x[x.index < T1].mean())
            res[f'ov_{period}_std_{col}'] = res.groupby(period)[col].transform(lambda x: x[x.index < T1].std())

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

    news = preprocess_news(asset=asset, keywords=keywords, use_weekends=use_weekends, last_day=last_day)
    mod = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    tmp = mod.encode(news['summary'], show_progress_bar=(last_day is None))
    embed_sum = pd.DataFrame(tmp, index=news.index)
    embed_sum.columns = [f'news_sum_{i}' for i in embed_sum]
    # embed = pd.concat([embed_title, embed_sum], axis=1).sort_index()
    return embed_sum.groupby(embed_sum.index).mean()


def sentiment_analysis(asset, keywords=None, use_weekends=True, last_day=None):
    """
    Loads a sentiment-analyzed, daily-aggregated version of a company's news.

    :param str asset: company's symbol (e.g. AAPL, MSFT)
    :param list[str] keywords: a list of keywords on which to filter the news to ensure relevance.
    :param bool use_weekends: whether news published during weekends should be used.
    :param None|str last_day: the last date before truncation, in format '%Y-%m-%d'. If None, no truncation is made.
    :return: A dataframe with each row corresponding to a pos/neg sentiment analysis, daily-aggregated company's news.
    :rtype: pd.DataFrame
    """

    news = preprocess_news(asset=asset, keywords=keywords, use_weekends=use_weekends, last_day=last_day)
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


def preprocess_news(asset, keywords=None, aggregate=False, use_weekends=True, last_day=None):
    """
    Preprocesses company's news (daily-aggregated) to be used in other functions.

    :param str asset: company's symbol (e.g. AAPL, MSFT)
    :param list[str] keywords: a list of keywords on which to filter the news to ensure relevance.
    :param bool aggregate: whether to daily-aggregate pieces of news.
    :param bool use_weekends: whether news published during weekends should be used.
    :param None|str last_day: the last date before truncation, in format '%Y-%m-%d'. If None, no truncation is made.
    :return: A dataframe with each row corresponding the daily-aggregated company's news.
    :rtype: pd.DataFrame
    """

    filename = '../data/intrinio/' + asset.lower() + '_news.csv'
    df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
    df.drop(['id', 'url', 'publication_date'], axis=1, inplace=True)
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

    for number, asset in enumerate(KEYWORDS.keys()):
        path = folder + f'{asset.lower()}_news_embed.csv'
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
        path = f'../data/intrinio/{asset.lower()}_news_embed.csv'
        embed = pd.read_csv(path, encoding='utf-8', index_col=0)
        embed = embed[embed.index < T1]
        df = pd.concat([df, embed], axis=0, ignore_index=True)
    pca = PCA(n_components=dim)
    pca.fit(df)
    joblib.dump(pca, f'../data/pca_sbert_{dim}.joblib')
    return pca
