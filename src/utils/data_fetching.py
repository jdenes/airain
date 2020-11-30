import pandas as pd
import requests

from .basics import clean_string


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
        df['date'] = df['publication_date'].str[:10]
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
