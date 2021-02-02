import pandas as pd
from utils.data_fetching import fetch_yahoo_intraday

companies = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'MSFT', 'IBM', 'CVX', 'JNJ', 'PG', 'PFE', 'VZ', 'BA', 'MRK',
             'CSCO', 'HD', 'MCD', 'MMM', 'GE', 'NKE', 'CAT', 'V', 'JPM', 'AXP', 'GS', 'UNH', 'TRV']


def fetch_yahoo_data_intraday():
    """
    Get data for intraday statistics.

    :rtype: None
    """
    folder = '../data/yahoo_intraday/'
    for company in companies:
        print(f'Fetching {company} data...')
        path = folder + company.lower()
        fetch_yahoo_intraday(filename=path + '_prices.csv', company=company)


def is_4_better_than_3():
    """
    Should you invest at 10:00 rather than 9:30?
    """
    folder = '../data/yahoo_intraday/'
    V1, V2 = [], []
    for company in companies:
        file = f'{folder}{company.lower()}_prices.csv'
        df = pd.read_csv(file, encoding='utf-8', index_col=0)
        df.index = df.index.rename('date')
        time_index = pd.to_datetime(df.index)
        df['unique_date'] = time_index.strftime('%Y-%m-%d')
        df['unique_time'] = time_index.strftime('%H:%M')
        df1 = []
        for date in df['unique_date'].unique():
            sub = df[df['unique_date'] == date]
            x = {'date': date,
                 '09:30': sub[sub.unique_time == '09:30']['open'].values[0],
                 '10:00': sub[sub.unique_time == '10:30']['open'].values[0],
                 '15:30': sub[sub.unique_time == '15:30']['open'].values[0],
                 '15:55': sub[sub.unique_time == '15:55']['close'].values[0]}
            df1.append(x)
        df1 = pd.DataFrame(df1).set_index('date')
        df1['label'] = (df1['15:55'] > df1['09:30']).astype(int)

        # If buy: it's better if 10:00 is LOWER than 09:30
        df1['diff_morning'] = (df1['10:00'] < df1['09:30']).astype(int)
        df1['diff_evening'] = (df1['15:30'] > df1['15:55']).astype(int)
        v1 = 100 * df1.groupby('label')['diff_morning'].mean().loc[1]
        v2 = 100 * df1.groupby('label')['diff_evening'].mean().loc[1]
        V1.append(v1), V2.append(v2)
        print(f'For {company} price at 10:00 is lower {v1:.2f}% and price at 15:30 is greater {v2:.2f}%')
    V1, V2 = sum(V1) / len(V1), sum(V2) / len(V2)
    print(f'Overall: buy latter is profitable {V1:.2f}%, sell earlier is profitable {V2:.2f}%')


if __name__ == "__main__":
    is_4_better_than_3()
