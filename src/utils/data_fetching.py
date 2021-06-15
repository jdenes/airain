import requests
import configparser
import pandas as pd
import yahooquery
from datetime import datetime
from .basics import write_data, clean_string


def fetch_intrinio_news(filename, api_key, company, update=False):
    """
    Fetches and saves Intrinio API to get assets' historical news.

    :param str filename: name of the file where to store the obtained data.
    :param str api_key: Intrinio personal API key.
    :param str company: company's symbol (e.g. AAPL, MSFT)
    :param bool update: whether you only want to update the data (gets only last 100 entries) or get all the history.
    :rtype: None
    """

    base_url = f'https://api-v2.intrinio.com/companies/{company}/news?page_size=100&api_key={api_key}'
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
        # df['date'] = df['publication_date'].str[:10]
        df['date'] = pd.to_datetime(df['publication_date'], format='%Y-%m-%dT%H:%M:%S.%fZ').dt.strftime('%Y-%m-%d')
        df['summary'] = df['summary'].apply(clean_string)
        df = df.set_index('date', drop=True)
        if next_page == 0 and not update:
            df.to_csv(filename, encoding='utf-8')
        else:
            df.to_csv(filename, encoding='utf-8', mode='a', header=False)
        next_page = data['next_page'] if not update else None
        url = base_url + f'&next_page={next_page}'
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

    base_url = f'https://api-v2.intrinio.com/securities/{company}/prices?page_size=100&api_key={api_key}'
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
        url = base_url + f'&next_page={next_page}'
    if update:
        df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        df.to_csv(filename, encoding='utf-8')


def fetch_poloniex_prices(filename, currency_pair, period=7200):
    """
    Fetches and saves Poloniex API to get currency pair historical prices.

    :param str filename: name of the file where to store the obtained data.
    :param str currency_pair: currency pair (e.g. USDT_BTC, BTC_ETH).
    :param int period: data frequency (one of 300, 900, 1800, 7200, 14400, 86400), default is 1800.
    :rtype: None
    """

    p = currency_pair.upper()
    e = datetime.now().timestamp()
    s = datetime(2019, 1, 1, 0, 0).timestamp()
    url = f"https://poloniex.com/public?command=returnChartData&currencyPair={p}&start={s}&end={e}&period={period}"
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.set_index('date', drop=True)
    write_data(filename, df)


def fetch_yahoo_news(filename, company):
    """
    Fetches and saves Yahoo API to get assets' historical news.

    :param str filename: name of the file where to store the obtained data.
    :param str company: company's symbol (e.g. AAPL, MSFT)
    :rtype: None
    """

    ticker = yahooquery.Ticker(company)
    df = ticker.news(100000)
    df = pd.DataFrame(df)
    df['date'] = pd.to_datetime(df['provider_publish_time'].astype(int), unit='s').dt.strftime('%Y-%m-%d')
    df['full_date'] = pd.to_datetime(df['provider_publish_time'].astype(int), unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.set_index('date', drop=True)
    df.drop(['rank', 'imageSet'], axis=1, inplace=True)
    df['tickers'] = df['tickers'].apply(lambda x: '|'.join(str(elt) for elt in x))
    write_data(filename, df, same_ids=True)


def fetch_yahoo_prices(filename, company):
    """
    Fetches and saves Yahoo API to get assets' historical prices.

    :param str filename: name of the file where to store the obtained data.
    :param str company: company's symbol (e.g. AAPL, MSFT)
    :rtype: None
    """

    ticker = yahooquery.Ticker(company)
    df = ticker.history(period='max')
    df.index = df.index.droplevel('symbol')
    df.index = pd.to_datetime(df.index.to_list()).strftime('%Y-%m-%d')
    write_data(filename, df)


def fetch_yahoo_intraday(filename, company):
    """
    Fetches and saves Yahoo API to get assets' past 2 months 5min frequency intraday prices.

    :param str filename: name of the file where to store the obtained data.
    :param str company: company's symbol (e.g. AAPL, MSFT)
    :rtype: None
    """

    ticker = yahooquery.Ticker(company)
    df = ticker.history(interval='5m', period='60d')
    df.index = df.index.droplevel('symbol')
    df.index = pd.to_datetime(df.index.to_list()).strftime('%Y-%m-%d %H:%M:%S')
    write_data(filename, df)


def fetch_intrinio_data(companies):
    """
    Fetches news and prices data for all companies.

    :param list companies: list of companies symbol for which to get data (e.g ['AAPL', 'MSFT'])
    :rtype: None
    """
    folder = '../data/intrinio/'
    parser = configparser.ConfigParser()
    parser.read('../resources/intrinio.cfg')
    api_key = parser['INTRINIO']['access_token']
    for company in companies:
        print(f'Fetching {company} data...')
        path = folder + company.lower()
        fetch_intrinio_news(filename=path + '_news.csv', api_key=api_key, company=company)
        fetch_intrinio_prices(filename=path + '_prices.csv', api_key=api_key, company=company)


def fetch_poloniex_data(pairs):
    """
    Fetches or updates prices data for all crypto pairs provided.

    :param list pairs: list of companies pair for which to get data (e.g ['BTC_ETH'])
    :return:
    """
    folder = '../data/poloniex/'
    for pair in pairs:
        path = folder + pair.lower()
        fetch_poloniex_prices(filename=path+'_prices.csv', currency_pair=pair)


def fetch_yahoo_data(companies):
    """
    Fetches or updates news and prices data for all companies provided.

    :param list companies: list of companies symbol for which to get data (e.g ['AAPL', 'MSFT'])
    :rtype: None
    """
    folder = '../data/yahoo/'
    for company in companies:
        print(f'Fetching {company} data...')
        path = folder + company.lower()
        # fetch_yahoo_news(filename=path + '_news.csv', company=company)
        fetch_yahoo_prices(filename=path + '_prices.csv', company=company)
        fetch_yahoo_intraday(f'../data/yahoo_intraday/{company.lower()}_prices.csv', company=company)
    fetch_yahoo_prices(filename=folder + '^n225_prices.csv', company='^n225')


def get_current_price(company):
    """
    Provides immediate asset price.

    :param str company: company's symbol (e.g. AAPL, MSFT)
    :rtype: None
    """

    ticker = yahooquery.Ticker(company)
    df = ticker.history(interval='1m', period='1d')
    df.index = df.index.droplevel('symbol')
    return df
