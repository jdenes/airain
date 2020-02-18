import time
import fxcmpy
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt

from traders import NeuralTrader, LstmTrader, ForestTrader, Dummy, Randommy, IdealTrader
from utils import load_data, fetch_crypto_rate, fetch_currency_rate, fetch_fxcm_data

freq = 1
h = 10
initial_gamble = 1000
amount = 2 * round(initial_gamble / (100 / 3))
fees = 0.0
alpha_key = "H2T4H92C43D9DT3D"
fxcm_key = '9c9f8a5725072aa250c8bd222dee004186ffb9e0'


def fetch_data():
    con = fxcmpy.fxcmpy(access_token=fxcm_key, server='demo')
    start, end = '2009-11-30 00:00:00', '2020-01-30 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_eurusd_train_5.csv', start=start, end=end, freq=freq, con=con)
    start, end = '2020-01-01 00:00:00', '2020-02-01 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_eurusd_test_5.csv', start=start, end=end, freq=freq, con=con)


def train_models():
    print('Training ASK model...')
    df, labels, price = load_data(filename='./data/dataset_eurusd_train_5.csv', target_col='askclose', shift=1)
    trader = ForestTrader(h=h)
    trader.ingest_traindata(df=df, labels=labels)
    print(trader.test(plot=True))
    trader.save(model_name='Huorn askclose 5')
    print('Training BID model...')
    df, labels, price = load_data(filename='./data/dataset_eurusd_train_5.csv', target_col='bidclose', shift=1)
    trader = ForestTrader(h=h)
    trader.ingest_traindata(df=df, labels=labels)
    print(trader.test(plot=True))
    trader.save(model_name='Huorn bidclose 5')


def backtest_models():
    df, labels, price = load_data(filename='./data/dataset_eurusd_test.csv', target_col='askclose', shift=1)
    ask_trader = ForestTrader(h=h)
    ask_trader.load(model_name='Huorn askclose')
    bid_trader = ForestTrader(h=h)
    bid_trader.load(model_name='Huorn bidclose')
    ask_backtest = ask_trader.backtest(df, labels, price, initial_gamble, fees)
    plt.plot(ask_backtest['value'], label='ASK model')
    bid_backtest = bid_trader.backtest(df, labels, price, initial_gamble, fees)
    plt.plot(bid_backtest['value'], label='BID model')
    baseline = Dummy().backtest(df, labels, price, initial_gamble, fees)
    plt.plot(baseline['value'], label='Pure USD')
    random = Randommy().backtest(df, labels, price, initial_gamble, fees)
    plt.plot(random['value'], label='Random')
    plt.legend()
    plt.grid()
    plt.show()


def mega_backtest():
    df, labels, price = load_data(filename='./data/dataset_eurusd_test.csv', target_col='askclose', shift=1)
    ask_trader, bid_trader = ForestTrader(h=h), ForestTrader(h=h)
    ask_trader.load(model_name='Huorn askclose')
    bid_trader.load(model_name='Huorn bidclose')
    order = None
    buy, sell = 0, 0
    buy_correct, sell_correct = 0, 0
    overall_profit = 0
    noth = 0
    profit_list = []

    X, _, ind = ask_trader.transform_data(df, labels, get_index=True)
    ask_preds = ask_trader.predict(X)
    bid_preds = bid_trader.predict(X)
    print(ind[0], df.index[h - 1], ind[-1], df.index[-1], len(X), len(df.index[h-1:]))

    for i in range(h - 1, len(df)):
        index = df.index[i - h + 1:i + 1]  # i-th should be included
        now_ask, now_bid = df.loc[index[-1]]['askclose'], df.loc[index[-1]]['bidclose']
        gpl = 0

        # Step one: close former position
        if order is not None:
            if order['is_buy']:
                pl = round((now_bid - order['open']) * order['amount'], 5)
                gpl = gross_pl(pl, amount, now_bid)
                buy_correct += int(now_bid > order['open'])
                buy += 1
            else:
                pl = round((order['open'] - now_ask) * order['amount'], 5)
                gpl = gross_pl(pl, amount, now_ask)
                sell_correct += int(now_ask < order['open'])
                sell += 1
        overall_profit = round(overall_profit + gpl, 2)
        profit_list.append(overall_profit)

        # Step two: open new position
        pred_ask, pred_bid = ask_preds[i - h + 1], bid_preds[i - h + 1]

        # if price is going up: buy
        if pred_bid > now_ask:
            order = {'is_buy': True, 'open': now_ask, 'exp_close': pred_bid, 'amount': amount}
        # elif price is going down: sell
        elif pred_ask < now_bid:
            order = {'is_buy': False, 'open': now_bid, 'exp_close': pred_ask, 'amount': amount}
        # else do nothing
        else:
            order = None
            noth += 1
        if i % 720 == 0:
            prog = round(100 * i / (len(df) - h))
            rien = round(100 * noth / (i - h + 1), 3)
            buy_acc = round(100 * buy_correct / buy, 3)
            sell_acc = round(100 * sell_correct / sell, 3)
            print('{}% ({}) | profit is {} and overall profit {} and correct share buy {} sell {} nothing {}'.format(
                prog, index[-1], gpl, overall_profit, buy_acc, sell_acc, rien)
            )

    plt.plot(profit_list, label='Profit evolution')
    plt.legend()
    plt.grid()
    plt.show()


def gross_pl(pl, K, price):
    return round((K / 10) * pl * (1 / price), 2)


def get_price_data(con):
    fetch_fxcm_data('./data/dataset_eurusd_now.csv', freq=freq, con=con, n_last=30)
    # print(con.get_prices('EUR/USD'))
    df, labels, price = load_data(filename='./data/dataset_eurusd_now.csv',
                                  target_col='askclose',
                                  shift=1,
                                  keep_last=True)
    return df, labels, price


def trade(con, trader, df, labels, price):
    res = trader.predict_next(df, labels, price, value=initial_gamble, fees=fees)
    # if going down: sell | if going up: buy
    is_buy = False if res['next_policy'] == (1, 0) else True
    con.open_trade(symbol='EUR/USD', is_buy=is_buy, amount=amount, time_in_force='GTC', order_type='AtMarket')
    return 'DOWN' if res['next_policy'] == (1, 0) else 'UP'


def heart_beat():
    t1 = datetime.now()
    count = 1
    con = fxcmpy.fxcmpy(access_token=fxcm_key, server='demo')
    # con.subscribe_market_data('EUR/USD')
    trader = ForestTrader(h=h)
    trader.load(model_name='Huorn askclose')
    print(datetime.now(), ': initialization took', datetime.now() - t1)

    while count < 3:
        now = datetime.now()
        if now.second == 0 and now.minute % freq == 0:
            t1 = datetime.now()
            print(datetime.now(), ': starting iteration', count)
            # con.close_all_for_symbol('EUR/USD')
            df, labels, price = get_price_data(con)
            print(datetime.now(), ': current price is', price.to_list()[-1])
            res = trade(con, trader, df, labels, price)
            print(datetime.now(), ': expected movement is', res)
            print(datetime.now(), ': iteration took', datetime.now() - t1)
            count += 1
        time.sleep(0.1)

    time.sleep(60 * freq)
    con.close_all_for_symbol('EUR/USD')
    print('Trading stopped.')


if __name__ == "__main__":
    # fetch_currency_rate('./data/dataset_eurgbp.csv', 'EUR', 'GBP', 5, alpha_key)
    # fetch_data()
    # train_models()
    # backtest_models()
    mega_backtest()

    # heart_beat()
