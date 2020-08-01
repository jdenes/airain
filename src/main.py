import os
import time
import fxcmpy
import configparser
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
from tqdm import tqdm

from traders import LstmTrader
from api_emulator import Emulator
from utils import fetch_fxcm_data, fetch_intrinio_news, fetch_intrinio_prices, append_data
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
config.read('../resources/intrinio.cfg')
api_key = config['INTRINIO']['access_token']
config.read('../resources/trading212.cfg')
user_name = config['TRADING212']['user_name']
pwd = config['TRADING212']['password']

target_col = 'close'

# Removed UTX
companies = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'MSFT', 'IBM', 'CVX', 'JNJ', 'PG', 'PFE', 'VZ', 'BA', 'MRK',
             'CSCO', 'HD', 'MCD', 'MMM', 'GE', 'NKE', 'CAT', 'V', 'JPM', 'AXP', 'GS', 'UNH', 'TRV']
leverages = {'AAPL': 5, 'XOM': 5, 'KO': 5, 'INTC': 5, 'WMT': 5, 'MSFT': 5, 'IBM': 5, 'CVX': 5, 'JNJ': 5,
             'PG': 5, 'PFE': 5, 'VZ': 5, 'BA': 5, 'MRK': 5, 'CSCO': 5, 'HD': 5, 'MCD': 5, 'MMM': 5,
             'GE': 5, 'NKE': 5, 'CAT': 5, 'V': 5, 'JPM': 5, 'AXP': 5, 'GS': 5, 'UNH': 5, 'TRV': 5}
performers = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'IBM', 'CVX', 'JNJ', 'PG', 'VZ', 'MRK', 'HD', 'GE', 'GS']


def fetch_intrinio_data():
    for company in companies:
        print('Fetching {} data...'.format(company))
        path = '../data/intrinio/{}'.format(company.lower())
        fetch_intrinio_news(filename=path+'_news.csv', api_key=api_key, company=company)
        fetch_intrinio_prices(filename=path+'_prices.csv', api_key=api_key, company=company)


def fetch_data():
    print('Fetching data...')
    con = fxcmpy.fxcmpy(config_file='../resources/fxcm.cfg', server='demo')
    start, end = '2002-01-01 00:00:00', '2020-01-01 00:00:00'
    fetch_fxcm_data(filename='../data/dataset_' + c + '_train_' + unit + f + '.csv',
                    curr=curr, unit=unit, start=start, end=end, freq=datafreq, con=con)
    start, end = '2020-01-01 00:00:00', '2020-06-01 00:00:00'
    fetch_fxcm_data(filename='../data/dataset_' + c + '_test_' + unit + f + '.csv',
                    curr=curr, unit=unit, start=start, end=end, freq=datafreq, con=con)


def train_model():
    print('Training model...')
    trader = LstmTrader(h=h, normalize=False)
    banks = [f[:-4] for f in os.listdir('../data/finance/') if f.endswith('.csv')]
    banks = companies

    df, labels = load_data('../data/intrinio/', tradefreq, datafreq)
    trader.ingest_traindata(df, labels)

    trader.train(epochs=epochs)
    trader.test(plot=False)
    trader.save(model_name='Huorn askopen NOW' + tf)


def backtest(plot=False):

    print('_' * 100, '\n')
    print('Initializing backtest...')
    trader = LstmTrader(load_from='Huorn askopen NOW' + tf)
    ov_df, ov_labels = load_data('../data/intrinio/', tradefreq, datafreq, start_from='2020-04-01')
    assets_profits, assets_returns = [], []

    for asset in enumerate(companies):
        if asset[1] in performers:
            print('_' * 100, '\n')
            print(f'Backtesting on {asset[1]}...\n')

            buy, sell, buy_correct, sell_correct, do_nothing = 0, 0, 0, 0, 0
            index_hist, balance_hist, returns_hist, pls_hist, bench_hist = [], [], [], [], []
            balance = bench_balance = initial_gamble

            df, labels = ov_df[ov_df['asset'] == asset[0]], ov_labels[ov_df['asset'] == asset[0]]
            X, P, y, ind = trader.transform_data(df, labels, get_index=True)
            df = df.loc[ind]

            preds = trader.predict(X, P)
            y_true = pd.Series(y.flatten())
            # y_pred = pd.Series(preds)
            # from sklearn.metrics import classification_report
            # print(classification_report(y_true, y_pred, digits=4))

            for i in range(1, len(df)):

                j, k = ind[i], ind[i-1]
                if dt.strptime(j, '%Y-%m-%d').minute % tradefreq == 0:

                    open, close = df.loc[j]['open'], df.loc[j]['close']
                    pl, gpl = 0, 0
                    # quantity = int(balance * 3 / 100)
                    amount = initial_gamble
                    quantity = int(amount * leverages[asset[1]] / open)

                    # Step 1 (morning) : decide what to do today and open position
                    pred = preds[i-1]
                    label = labels[ind[i-1]]
                    order = decide_order(asset[1], quantity, open, pred, j)

                    # Step 2 (evening): close position
                    if order['is_buy'] is not None:
                        if order['is_buy']:
                            pl = round((close - order['open']) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)
                            buy_correct += int(pl >= 0)
                            buy += 1
                        else:
                            pl = round((order['open'] - close) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)
                            sell_correct += int(pl >= 0)
                            sell += 1
                    else:
                        do_nothing += 1

                    # Step 3 bis: compute metrics and stuff
                    balance, returns, pls = round(balance + pl, 2), round((pl / amount) * 100, 2), round(pl, 2)
                    bench_balance = round(bench_balance + (close - order['open']) * quantity, 2)
                    balance_hist.append(balance), returns_hist.append(returns), pls_hist.append(pls)
                    bench_hist.append(bench_balance)
                    index_hist.append(ind[i])

            pos_pls, neg_pls = [i for i in pls_hist if i > 0], [i for i in pls_hist if i < 0]
            profit = round(balance - initial_gamble, 2)
            bench = round(bench_balance - initial_gamble, 2)
            buy_acc = round(100 * buy_correct / buy, 2) if buy != 0 else 'NA'
            sell_acc = round(100 * sell_correct / sell, 2) if sell != 0 else 'NA'
            pos_days = round(100 * sum([i >= 0 for i in pls_hist]) / len(pls_hist), 2)
            m_returns = round(sum(returns_hist) / len(returns_hist), 2)
            m_wins = round(sum(pos_pls) / len(pos_pls), 2) if len(pos_pls) != 0 else 'NA'
            m_loss = round(sum(neg_pls) / len(neg_pls), 2) if len(neg_pls) != 0 else 'NA'

            assets_profits.append(profit), assets_returns.append(m_returns)
            print(f'Profit: {profit}. Benchmark: {bench}. Mean daily return: {m_returns}%.')
            print(f'Correct moves: {pos_days}%. Correct buy: {buy_acc}%. Correct sell: {sell_acc}%.')
            print(f'Av. wins/loss amounts: {m_wins}/{m_loss}. Ext. wins/loss amounts: {max(pls_hist)}/{min(pls_hist)}.')
            if plot:
                nice_plot(index_hist, [balance_hist, bench_hist], ['Algorithm', 'Benchmark'],
                          title=f'Profit evolution for {asset[1]}')

    print('_' * 100, '\n')
    n_pos = len([x for x in assets_profits if x > 0])
    m_prof = round(sum(assets_profits) / len(assets_profits), 2)
    m_ret = round(sum(assets_returns) / len(assets_returns), 2)
    print('Returns:', assets_returns)
    print(f'Average profit by assets: {m_prof}. Average daily return: {m_ret}%.')
    print(f'Profitable assets: {n_pos}/{len(assets_profits)}.')
    print('_' * 100, '\n')
    # print([co for i, co in enumerate(companies) if profits[i] > initial_gamble/2])


def decide_order(asset, quantity, open, pred, date):
    # if pred_bid > (1 - tolerance) * now_ask:
    if pred == 1:
        order = {'asset': asset, 'is_buy': True, 'open': open, 'quantity': quantity, 'date': date}
    # elif pred_ask < now_bid / (1 - tolerance):
    elif pred == 0:
        order = {'asset': asset, 'is_buy': False, 'open': open, 'quantity': quantity, 'date': date}
    else:
        order = {'asset': asset, 'is_buy': None, 'open': open, 'quantity': quantity, 'date': date}
    return order


def gross_pl(pl, K, price):
    return round((K / 10) * pl * (1 / price), 2)


def get_yesterday_perf():
    
    print('_' * 100, '\n')
    print("Correctness of yesterday's predictions")
    
    df, _ = load_data('../data/intrinio/', tradefreq, datafreq)
    reco = pd.read_csv('../resources/recommendations.csv', encoding='utf-8', index_col=0)
    prices = pd.read_csv('../resources/open_prices.csv', encoding='utf-8', index_col=0)
    
    date, col_names = reco.iloc[-1].name, reco.columns
    df = df[df['asset'].isin([i for i, co in enumerate(companies) if co in col_names])].loc[date].reset_index(drop=True)
    reco = reco.iloc[-2].reset_index(drop=True)
    prices = prices[col_names].loc[date].reset_index(drop=True)
    quantity = (initial_gamble * pd.Series([leverages[co] for co in col_names]) / df['open']).astype(int)
    accuracy = (reco == (df['close'] > df['open']))
    profits = (2 * reco - 1) * (df['close'] - df['open']) * quantity

    print("Computed with data from {}, traded on {}.".format(reco.name, date))
    print('_' * 100, '\n')
    print('Asset | Quantity | Reco | Profit/Loss | Exp Open | Act Open | Exp Close')
    print('-'*71)
    for i, x in enumerate(profits):
        print('{:5s} | {:8d} | {:4d} | {:11.2f} | {:8.2f} | {:8.2f} | {:9.2f}'.format(
              col_names[i], quantity[i], reco[i], x, df['open'][i], prices[i], df['close'][i]))
    print('_' * 100, '\n')
    print('Accuracy was {:.2f}%. Total P/L was {:.2f}.'.format(100 * accuracy.mean(), profits.sum()))


def update_data():
    folder = '../data/intrinio/'
    for company in companies:
        print('Fetching most recent {} data...'.format(company))
        path = folder + company.lower()
        fetch_intrinio_news(filename=path+'_news.csv', api_key=api_key, company=company, update=True)
        fetch_intrinio_prices(filename=path+'_prices.csv', api_key=api_key, company=company, update=True)
    load_data(folder, tradefreq, datafreq, update_embed=True)


def get_recommendations():
    print('_' * 100, '\n')
    now = time.time()
    trader = LstmTrader(load_from='Huorn askopen NOW' + tf)
    df, labels = load_data('../data/intrinio/', tradefreq, datafreq)
    yesterday = df.index.max()
    df = df.loc[yesterday].reset_index(drop=True)
    X, P, _, ind = trader.transform_data(df, labels, get_index=True)
    preds = trader.predict(X, P)
    reco, order_book = {'date': yesterday}, []
    lev = pd.Series([leverages[co] for co in companies])
    quantity = (initial_gamble * lev / df['close']).astype(int)
    print('On {}, predictions for next day are:'.format(yesterday))
    print('_' * 100, '\n')
    print('Asset | Quantity | Reco')
    print('-'*24)
    for i, pred in enumerate(preds):
        if companies[i] in performers:
            reco[companies[i]] = pred
            order_book.append({'asset': companies[i], 'is_buy': pred, 'quantity': int(quantity[i])})
            print('{:5s} | {:8d} | {:4d}'.format(companies[i], quantity[i], pred))
    path = '../resources/recommendations.csv'
    reco = pd.DataFrame([reco]).set_index('date', drop=True)
    append_data(path, reco)
    print('Inference took {} seconds.'.format(round(time.time()-now)))
    return order_book


def place_orders(order_book):
    emulator = Emulator(user_name, pwd)
    emulator.close_all_trades()
    for order in order_book:
        emulator.open_trade(order)
    prices = emulator.get_open_prices()
    prices = pd.DataFrame([prices]).set_index('date', drop=True)
    path = '../resources/open_prices.csv'
    append_data(path, prices)
    time.sleep(60)
    emulator.quit()


if __name__ == "__main__":
    # fetch_intrinio_data()
    train_model()
    backtest(plot=False)
    # update_data()
    # order_book = get_recommendations()
    # place_orders(order_book)
    # get_yesterday_perf()

    # emulator = Emulator(user_name, pwd)
    # prices = emulator.get_open_prices()
    # prices = emulator.get_close_prices()
    # emulator.quit()
