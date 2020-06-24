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
initial_gamble = 1000
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
performers = ['AAPL', 'KO', 'INTC', 'WMT', 'MSFT', 'IBM', 'PG', 'PFE', 'VZ', 'MRK',
              'CSCO', 'HD', 'MCD', 'GE', 'NKE', 'CAT', 'V', 'JPM', 'GS', 'UNH']


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
    trader = LstmTrader(h=h, normalize=True)
    
    # filename = './data/dataset_{}_train_{}{}.csv'.format(c, unit, f)
    banks = [f[:-4] for f in os.listdir('./data/finance/') if f.endswith('.csv')]
    banks = companies

    df, labels = load_data('./data/intrinio/', tradefreq, datafreq)
    trader.ingest_traindata(df, labels)

    trader.train(epochs=epochs)
    trader.test(plot=False)
    trader.save(model_name='Huorn askopen NOW' + tf)


def mega_backtest():

    print('_' * 100, '\n')
    print('Initializing backtest...')
    trader = LstmTrader(load_from='Huorn askopen NOW' + tf)
    ov_df, ov_labels = load_data('./data/intrinio/', tradefreq, datafreq, start_from='2020-01-01')
    profits = []
    do_plot = False

    for asset in enumerate(companies):
        if asset[1] in performers:
            print('_' * 100, '\n')
            print('Backtesting on {}...'.format(asset[1]))

            buy, sell, buy_correct, sell_correct, do_nothing = 0, 0, 0, 0, 0
            index_list, profit_list, benchmark = [], [], []
            order = {'is_buy': None}
            count = 1
            balance = bench_balance = initial_gamble

            df, labels = ov_df[ov_df['asset'] == asset[0]], ov_labels[ov_df['asset'] == asset[0]]
            X, P, y, ind = trader.transform_data(df, labels, get_index=True)
            df = df.loc[ind]

            preds = trader.predict(X, P)
            y_true = pd.Series(y.flatten())
            y_pred = pd.Series(preds)
            from sklearn.metrics import classification_report
            print(classification_report(y_true, y_pred, digits=4))

            for i in range(lag, len(df)-1):

                j, k = ind[i], ind[i+1]
                if dt.strptime(j, '%Y-%m-%d').minute % tradefreq == 0:

                    now_close, now_open = df.loc[j]['close'], df.loc[j]['open']
                    fut_close, fut_open = df.loc[k]['close'], df.loc[k]['open']
                    pl, gpl = 0, 0
                    # amount = int(balance * 3 / 100)
                    amount = initial_gamble
                    amount = round(amount / now_open, 5)

                    # Step one: close former position if needed, else continue
                    if order['is_buy'] is not None:
                        if order['is_buy']:
                            pl = round((now_close - order['open']) * order['amount'], 5)
                            # gpl = gross_pl(pl, amount, now_close)
                            buy_correct += int(pl > 0)
                            buy += 1
                        else:
                            pl = round((order['open'] - now_close) * order['amount'], 5)
                            # gpl = gross_pl(pl, amount, now_close)
                            sell_correct += int(pl > 0)
                            sell += 1
                    else:
                        do_nothing += 1

                    # Step one bis: compute metrics and stuff
                    balance = round(balance + pl, 2)
                    profit_list.append(balance)
                    if count > 1:
                        bench_balance = round(bench_balance + (now_close - order['open']) * amount, 2)
                    benchmark.append(bench_balance)
                    index_list.append(ind[i])
                    count += 1

                    # Step two: decide what to do next
                    pred = preds[i - lag]
                    label = labels[ind[i - lag]]
                    order = decide_order(amount, fut_open, fut_close, pred)

            profit = round(balance - initial_gamble, 2)
            no_trade = round(100 * do_nothing / len(index_list), 2)
            buy_acc = round(100 * buy_correct / buy, 2) if buy != 0 else 'NA'
            sell_acc = round(100 * sell_correct / sell, 2) if sell != 0 else 'NA'

            profits.append(profit)
            print('Overall profit: {}. Correct buy: {}%. Proportion of hold: {}%.'.format(profit, buy_acc, no_trade))
            if do_plot:
                nice_plot(index_list, [profit_list, benchmark], ['Profit evolution', 'Benchmark'],
                          title='Equity evolution for ' + str(asset[1]))

    print('_' * 100, '\n')
    n_pos = len([x for x in profits if x > 0])
    m_prof = round(sum(profits) / len(profits), 2)
    print('Average profit across assets: {}. Number of profitable assets: {}/{}.'.format(m_prof, n_pos, len(profits)))
    print('_' * 100, '\n')
    # print([c for i, c in enumerate(companies) if profits[i] > 50])


def decide_order(amount, fut_open, fut_close, pred):
    # if price is going up: buy
    # if pred_bid > (1 - tolerance) * now_ask:
    if pred == 1:
        order = {'is_buy': True, 'open': fut_open, 'exp_close': fut_close, 'amount': amount}
    # elif price is going down: sell
    # elif pred_ask < now_bid / (1 - tolerance):
    elif pred == 0:
        order = {'is_buy': None, 'open': fut_open, 'exp_close': fut_close, 'amount': amount}
    # else do nothing
    else:
        order = {'is_buy': None, 'open': fut_open, 'exp_close': fut_close, 'amount': amount}
    return order


def gross_pl(pl, K, price):
    return round((K / 10) * pl * (1 / price), 2)


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


def get_current_askbid(con):
    data = con.get_prices('EUR/USD')
    # t1 = dt.now()
    # t1 = t1 + timedelta(hours=1) - timedelta(seconds=t1.second, microseconds=t1.microsecond)
    # data = data.truncate(after=t1)
    # print(data.index[0])
    data = data.tail(n=1).mean(axis=0)
    now_ask = data['Ask']
    now_bid = data['Bid']
    return now_ask, now_bid


def get_balance(con):
    data = con.get_accounts()
    data = data[data['accountId'] == account_id]
    return data['balance'].values[0]


def trade(con, order, amount):
    is_buy = order['is_buy']
    if is_buy is not None:
        con.open_trade(symbol='EUR/USD', is_buy=is_buy, amount=amount, time_in_force='GTC', order_type='AtMarket')


def get_next_preds():
    now = time.time()
    trader = LstmTrader(load_from='Huorn askopen NOW' + tf)
    df, labels = get_last_data()
    yesterday = df.index.max()
    df = df.loc[yesterday]
    X, P, _, _ = trader.transform_data(df, labels, get_index=True)
    preds = trader.predict(X, P)
    print('On {}, predictions for next day are:'.format(yesterday))
    for i, pred in enumerate(preds):
        print('{:<5s}: {}'.format(companies[i], pred))
    print('Inference took {} sec'.format(round(time.time()-now)))


if __name__ == "__main__":
    # fetch_currency_rate('./data/dataset_eurgbp.csv', 'EUR', 'GBP', 5, alpha_key)
    # fetch_data()
    # fetch_intrinio_data()
    # train_models()
    # mega_backtest()
    get_next_preds()

    # res = heart_beat()
