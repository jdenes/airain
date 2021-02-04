
def write_data(path, new_row, same_ids=False):
    """
    Writes data as .csv by creating a file or adding to an existing one if relevant.

    :param str path: path of the file where to write the data.
    :param pd.DataFrame new_row: dataframe to write.
    :param bool same_ids: whether the data contains duplicated indexes.
    :rtype: None
    """
    import os
    import pandas as pd
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


def clean_string(string):
    """
    Removes HTML tags, URLs, PDFs, brackets, tickers in parenthesis, multiple spaces and line breaks from string.

    :param str string: string to clean.
    :return: cleaned string.
    :rtype: str
    """
    import re
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'<[^<]+?>', '', string)
    string = re.sub(r'\[.+?\]', '', string)
    string = re.sub(r'\S +\.pdf', '', string)
    string = re.sub(r'http[s]?://\S+', '', string)
    # string = re.sub(r'\([A-Z]+:?\s?[A-Z]*\)', '', string)
    string = re.sub(r'\s+', ' ', string)
    return string.strip() + ' '


def normalize_data(data, data_max, data_min):
    """
    Normalizes data using min-max normalization.

    :param int|float|pd.Series data: data array or scalar to un-normalize.
    :param int|float data_max: max value to use in min-max normalization.
    :param int|float data_min: data_max: min value to use in min-max normalization.
    :return: un-normalized data.
    :rtype: int|float|pd.Series
    """

    return (data - data_min) / (data_max - data_min)


def unnormalize_data(data, data_max, data_min):
    """
    Un-normalizes data using min-max normalization.

    :param int|float|pd.Series data: data array or scalar to un-normalize.
    :param int|float data_max: max value used in min-max normalization.
    :param int|float data_min: data_max: min value used in min-max normalization.
    :return: un-normalized data.
    :rtype: int|float|pd.Series
    """

    return data * (data_max - data_min) + data_min


def omega2assets(value, omega, prices):
    """
    From a portfolio desired value, a pie (omega), and assets' prices, returns portfolio in term of asset numbers.

    :param int|float value: desired value of the portfolio.
    :param np.array omega: assets share of the portfolio (must sum up to one).
    :param np.array prices: buying price of each asset.
    :return: a portfolio in term of assets number.
    :rtype: np.array
    """
    import numpy as np
    return np.floor(np.array(value * omega).round(2) / prices)


def evaluate_portfolio(portfolio, prices):
    """
    Get the value of a portfolio given its assets prices.

    :param np.array portfolio: portfolio, each component being the number of each asset.
    :param prices: corresponding price of each asset.
    :return: value of portfolio.
    :rtype: float
    """
    import numpy as np
    return np.sum(portfolio * prices).round(2)
