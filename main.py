import os
import time
import fxcmpy
import configparser
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
from tqdm import tqdm

from traders import LstmTrader
from utils import fetch_fxcm_data, fetch_intrinio_news, fetch_intrinio_prices, merge_finance_csv
from utils import load_data, nice_plot, change_accuracy, normalize_data

unit = 'H'  # 'm' ou 'd'
datafreq = 1
tradefreq = 1
f, tf = str(datafreq), str(tradefreq)
lag = 0
h = 30
initial_gamble = 2000
fees = 0.0
tolerance = 0e-5  # 2
epochs = 50

curr = 'EUR/USD'
c = 'eurusd'

config = configparser.ConfigParser()
config.read('./resources/alphavantage.cfg')
api_key = config['ALPHAVANTAGE']['access_token']
config.read('./resources/fxcm.cfg')
account_id = config['FXCM']['account_id']
config.read('./resources/intrinio.cfg')
api_key = config['INTRINIO']['access_token']

target_col = 'close'

companies = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'MSFT', 'IBM', 'CVX', 'JNJ', 'PG', 'PFE', 'VZ', 'BA', 'MRK',
             'CSCO', 'HD', 'MCD', 'MMM', 'GE', 'UTX', 'NKE', 'CAT', 'V', 'JPM', 'AXP', 'GS', 'UNH', 'TRV'] 
performers = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'IBM', 'CVX', 'JNJ', 'PG', 'VZ', 'MRK', 'HD', 'GE', 'GS']


def fetch_intrinio_data():
    for company in companies:
        print('Fetching {} data...'.format(company))    
        path = './data/intrinio/{}'.format(company.lower())
        fetch_intrinio_news(filename=path+'_news.csv', api_key=api_key, company=company)
        fetch_intrinio_prices(filename=path+'_prices.csv', api_key=api_key, company=company)


def fetch_data():
    print('Fetching data...')
    con = fxcmpy.fxcmpy(config_file='./resources/fxcm.cfg', server='demo')
    start, end = '2002-01-01 00:00:00', '2020-01-01 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_' + c + '_train_' + unit + f + '.csv',
                    curr=curr, unit=unit, start=start, end=end, freq=datafreq, con=con)
    start, end = '2020-01-01 00:00:00', '2020-06-01 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_' + c + '_test_' + unit + f + '.csv',
                    curr=curr, unit=unit, start=start, end=end, freq=datafreq, con=con)


def train_models():
    print('Training model...')
    trader = LstmTrader(h=h, normalize=False)
    
    # filename = './data/dataset_{}_train_{}{}.csv'.format(c, unit, f)
    banks = [f[:-4] for f in os.listdir('./data/finance/') if f.endswith('.csv')]
    banks = companies

    df, labels = load_data('./data/intrinio/', tradefreq, datafreq)
    trader.ingest_traindata(df, labels)

    trader.train(epochs=epochs)
    trader.test(plot=False)
    trader.save(model_name='Huorn askopen NOW' + tf)


def mega_backtest(plot=False):

    print('_' * 100, '\n')
    print('Initializing backtest...')
    trader = LstmTrader(load_from='Huorn askopen NOW' + tf)
    ov_df, ov_labels = load_data('./data/intrinio/', tradefreq, datafreq, start_from='2020-04-01')
    profits = []

    for asset in enumerate(companies):
        if asset[1] in performers:
            print('_' * 100, '\n')
            print('Backtesting on {}...'.format(asset[1]))

            buy, sell, buy_correct, sell_correct, do_nothing = 0, 0, 0, 0, 0
            index_list, profit_list, benchmark = [], [], []
            balance = bench_balance = initial_gamble

            df, labels = ov_df[ov_df['asset'] == asset[0]], ov_labels[ov_df['asset'] == asset[0]]
            X, P, y, ind = trader.transform_data(df, labels, get_index=True)
            df = df.loc[ind]

            preds = trader.predict(X, P)
            y_true = pd.Series(y.flatten())
            y_pred = pd.Series(preds)
            from sklearn.metrics import classification_report
            print(classification_report(y_true, y_pred, digits=4))

            for i in range(lag, len(df)):

                j = ind[i]
                if dt.strptime(j, '%Y-%m-%d').minute % tradefreq == 0:

                    open, close = df.loc[j]['open'], df.loc[j]['close'],
                    pl, gpl = 0, 0
                    # quantity = int(balance * 3 / 100)
                    amount = initial_gamble
                    quantity = int(amount * 20 / open)

                    # Step 1 (morning) : decide what to do today and open position
                    pred = preds[i - lag]
                    label = labels[ind[i - lag]]
                    order = decide_order(quantity, open, pred, j)

                    # Step 2 (evening): close position
                    if order['is_buy'] is not None:
                        if order['is_buy']:
                            pl = round((close - order['open']) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)
                            buy_correct += int(pl > 0)
                            buy += 1
                        else:
                            pl = round((order['open'] - close) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)
                            sell_correct += int(pl > 0)
                            sell += 1
                    else:
                        do_nothing += 1

                    # Step 3 bis: compute metrics and stuff
                    balance = round(balance + pl, 2)
                    profit_list.append(balance)
                    bench_balance = round(bench_balance + (close - order['open']) * quantity, 2)
                    benchmark.append(bench_balance)
                    index_list.append(ind[i])

            profit = round(balance - initial_gamble, 2)
            bench = round(bench_balance - initial_gamble, 2)
            no_trade = round(100 * do_nothing / len(index_list), 2)
            buy_acc = round(100 * buy_correct / buy, 2) if buy != 0 else 'NA'
            sell_acc = round(100 * sell_correct / sell, 2) if sell != 0 else 'NA'

            profits.append(profit)
            print('Profit: {}. Bench: {}. Correct buy: {}%. Correct sell: {}%. Holds: {}%.'.format(
                  profit, bench, buy_acc, sell_acc, no_trade))
            if plot:
                nice_plot(index_list, [profit_list, benchmark], ['Algorithm', 'Benchmark'],
                          title='Profit evolution for ' + str(asset[1]))

    print('_' * 100, '\n')
    n_pos = len([x for x in profits if x > 0])
    m_prof = round(sum(profits) / len(profits), 2)
    print('Average profit across assets: {}. Number of profitable assets: {}/{}.'.format(m_prof, n_pos, len(profits)))
    print('_' * 100, '\n')
    # print([co for i, co in enumerate(companies) if profits[i] > initial_gamble/2])


def decide_order(quantity, open, pred, date):
    # if price is going up: buy
    # if pred_bid > (1 - tolerance) * now_ask:
    if pred == 1:
        order = {'is_buy': True, 'open': open, 'quantity': quantity, 'date': date}
    # elif price is going down: sell
    # elif pred_ask < now_bid / (1 - tolerance):
    elif pred == 0:
        order = {'is_buy': False, 'open': open, 'quantity': quantity, 'date': date}
    # else do nothing
    else:
        order = {'is_buy': None, 'open': open, 'quantity': quantity, 'date': date}
    return order


def gross_pl(pl, K, price):
    return round((K / 10) * pl * (1 / price), 2)


def get_yesterday_accuracy():
    print('_' * 100, '\n')
    folder = './data/intrinio/'
    df, _ = load_data(folder, tradefreq, datafreq)
    reco = pd.read_csv('./resources/recommendations.csv', encoding='utf-8', index_col=0)
    col_names = reco.columns
    ind = df['asset'].isin([i for i, co in enumerate(companies) if co in col_names])
    df = df[ind]
    date = reco.iloc[-1].name
    reco = reco.iloc[-2].reset_index(drop=True)
    df = df.loc[date].reset_index(drop=True)
    labels = df['close'] > df['open']
    quantity = (initial_gamble * 20 / df['open']).astype(int)
    print("Correctness of yesterday's predictions")
    print("Computed with data from {}, traded on {}.".format(reco.name, date))
    print('_' * 100, '\n')
    accuracy = (reco == labels)
    profits = (2 * reco - 1) * (df['close'] - df['open']) * quantity
    print('Asset | Quantity | Reco | Profit/Loss |    Open |   Close')
    print('-'*57)
    for i, x in enumerate(profits):
        print('{:5s} | {:8d} | {:4d} | {:11.2f} | {:7.2f} | {:7.2f}'.format(
              col_names[i], quantity[i], reco[i], x, df['open'][i], df['close'][i]))
    print('_' * 100, '\n')
    print('Accuracy was {:.2f}%. Total P/L was {:.2f}.'.format(100 * accuracy.mean(), profits.sum()))


def get_last_data(con=None):
    # fetch_fxcm_data('./data/dataset_' + c + '_now_' + f + '.csv', curr=curr, freq=datafreq, con=con, n_last=30)
    folder = './data/intrinio/'
    for company in companies:
        print('Fetching most recent {} data...'.format(company))
        path = folder + company.lower()
        fetch_intrinio_news(filename=path+'_news.csv', api_key=api_key, company=company, update=True)
        fetch_intrinio_prices(filename=path+'_prices.csv', api_key=api_key, company=company, update=True)
    df, labels = load_data(folder, tradefreq, datafreq, keep_last=True)
    return df, labels


def get_next_preds():
    now = time.time()
    trader = LstmTrader(load_from='Huorn askopen NOW' + tf)
    df, labels = get_last_data()
    yesterday = df.index.max()
    df = df.loc[yesterday]
    X, P, _, ind = trader.transform_data(df, labels, get_index=True)
    preds = trader.predict(X, P)
    res = {'date': yesterday}
    quantity = (initial_gamble * 20 / df['close']).round()
    print('On {}, predictions for next day are:'.format(yesterday))
    for i, pred in enumerate(preds):
        if companies[i] in performers:
            res[companies[i]] = pred
            print('{:<5s}: {}, quantity: {:.0f}'.format(companies[i], pred, quantity[i]))
    path = './resources/recommendations.csv'
    res = pd.DataFrame([res]).set_index('date', drop=True)
    df = pd.read_csv(path, encoding='utf-8', index_col=0)
    df = pd.concat([df, res], axis=0)
    df = df.loc[~df.index.duplicated(keep='last')]
    df.to_csv(path, encoding='utf-8')
    print('Inference took {} seconds.'.format(round(time.time()-now)))


if __name__ == "__main__":
    # fetch_data()
    # fetch_intrinio_data()
    # train_models()
    # mega_backtest(plot=False)
    get_next_preds()
    get_yesterday_accuracy()
