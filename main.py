import time
import fxcmpy
import pandas as pd
from datetime import datetime as dt

from traders import NeuralTrader, LstmTrader, ForestTrader, Dummy, Randommy, IdealTrader
from utils import load_data, fetch_crypto_rate, fetch_currency_rate, fetch_fxcm_data, nice_plot

freq = 1
f = str(freq)
h = 10
initial_gamble = 10000
fees = 0.0
tolerance = 4e-6

account_id = '1195258'
alpha_key = "H2T4H92C43D9DT3D"
fxcm_key = '9c9f8a5725072aa250c8bd222dee004186ffb9e0'


def fetch_data():
    con = fxcmpy.fxcmpy(access_token=fxcm_key, server='demo')
    start, end = '2005-11-30 00:00:00', '2020-01-30 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_eurusd_train_' + f + '.csv', start=start, end=end, freq=freq, con=con)
    start, end = '2020-01-01 00:00:00', '2020-02-01 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_eurusd_test_' + f + '.csv', start=start, end=end, freq=freq, con=con)


def train_models():
    print('Training ASK model...')
    df, labels, price = load_data(filename='./data/dataset_eurusd_train_' + f + '.csv', target_col='askclose', shift=1)
    trader = ForestTrader(h=h)
    trader.ingest_traindata(df=df, labels=labels)
    trader.train(n_estimators=100)
    print(trader.test(plot=True))
    trader.save(model_name='Huorn askclose ' + f)
    print('Training BID model...')
    df, labels, price = load_data(filename='./data/dataset_eurusd_train_' + f + '.csv', target_col='bidclose', shift=1)
    trader = ForestTrader(h=h)
    trader.ingest_traindata(df=df, labels=labels)
    trader.train(n_estimators=100)
    print(trader.test(plot=True))
    trader.save(model_name='Huorn bidclose ' + f)


def backtest_models():

    curves, names = [], []
    df, labels, price = load_data(filename='./data/dataset_eurusd_test_' + f + '.csv', target_col='askclose', shift=1)
    ask_trader = ForestTrader(h=h)
    ask_trader.load(model_name='Huorn askclose ' + f)
    ask_backtest = ask_trader.backtest(df, labels, price, initial_gamble, fees)
    curves.append(ask_backtest['value']), names.append('ASK model')

    df, labels, price = load_data(filename='./data/dataset_eurusd_test_' + f + '.csv', target_col='bidclose', shift=1)
    bid_trader = ForestTrader(h=h)
    bid_trader.load(model_name='Huorn bidclose ' + f)
    bid_backtest = bid_trader.backtest(df, labels, price, initial_gamble, fees)
    curves.append(bid_backtest['value']), names.append('BID model')

    baseline = Dummy().backtest(df, labels, price, initial_gamble, fees)
    curves.append(baseline['value']), names.append('Pure USD')
    random = Randommy().backtest(df, labels, price, initial_gamble, fees)
    curves.append(random['value']), names.append('Random')

    nice_plot(ind=baseline['index'], curves_list=curves, names=names, title='Equity evolution, no spread, f = ' + f)


def mega_backtest():
    df, labels, price = load_data(filename='./data/dataset_eurusd_test_' + f + '.csv', target_col='askclose', shift=1)
    ask_trader, bid_trader = ForestTrader(h=h), ForestTrader(h=h)
    ask_trader.load(model_name='Huorn askclose ' + f)
    bid_trader.load(model_name='Huorn bidclose ' + f)
    order = {'is_buy': None}
    buy, sell, buy_correct, sell_correct, do_nothing = 0, 0, 0, 0, 0
    profit_list = []

    balance = initial_gamble

    X, _, ind = ask_trader.transform_data(df, labels, get_index=True)
    ask_preds = ask_trader.predict(X)
    bid_preds = bid_trader.predict(X)
    # ask_preds = df.loc[ind]['askclose'].shift(-1).to_list()
    # bid_preds = df.loc[ind]['bidclose'].shift(-1).to_list()

    for i in range(h - 1, len(df)):
        index = df.index[i - h + 1:i + 1]  # i-th should be included
        now_ask, now_bid = df.loc[df.index[i]]['askclose'], df.loc[df.index[i]]['bidclose']
        gpl = 0
        amount = int(balance * 3 / 100)

        # Step one: close former position
        if order['is_buy'] is not None:
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
        else:
            do_nothing += 1

        balance = round(balance + gpl, 2)
        profit_list.append(balance)

        # Step two: open new position
        now_ask, now_bid = df.loc[df.index[i]]['askopen'], df.loc[df.index[i]]['bidopen']
        pred_ask, pred_bid = ask_preds[i - h + 1], bid_preds[i - h + 1]
        order = decide_order(amount, now_bid, now_ask, pred_bid, pred_ask)

    rien = round(100 * do_nothing / len(range(h - 1, len(df))), 3)
    buy_acc = round(100 * buy_correct / buy, 3)
    sell_acc = round(100 * sell_correct / sell, 3)

    print('Overall profit: {} | Correct share buy {} | Correct share sell {} | Share of done nothing {}'.format(
          round(balance - initial_gamble, 2), buy_acc, sell_acc, rien)
          )
    nice_plot(ind, [profit_list], ['Profit evolution'], title='Equity evolution, with spread, f = ' + f)


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


def get_current_askbid(con):
    data = con.get_prices('EUR/USD')
    now_ask = data['Ask'].to_list()[-1]
    now_bid = data['Bid'].to_list()[-1]
    return now_ask, now_bid


def get_balance(con):
    data = con.get_accounts()
    data = data[data['accountId'] == account_id]
    return data['balance'].values[0]


def decide_order(amount, now_bid, now_ask, pred_bid, pred_ask):
    # if price is going up: buy
    if pred_bid > (1 - tolerance) * now_ask:
        order = {'is_buy': True, 'open': now_ask, 'exp_close': pred_bid, 'amount': amount}
    # elif price is going down: sell
    elif pred_ask < now_bid / (1 - tolerance):
        order = {'is_buy': False, 'open': now_bid, 'exp_close': pred_ask, 'amount': amount}
    # else do nothing
    else:
        order = {'is_buy': None, 'open': now_bid, 'exp_close': pred_ask, 'amount': amount}
    return order


def trade(con, order, amount):
    is_buy = order['is_buy']
    if is_buy is not None:
        con.open_trade(symbol='EUR/USD', is_buy=is_buy, amount=amount, time_in_force='GTC', order_type='AtMarket')


def heart_beat():
    t1 = dt.now()
    count = 1
    con = fxcmpy.fxcmpy(access_token=fxcm_key, server='demo')
    con.subscribe_market_data('EUR/USD')
    old_balance = get_balance(con)

    ask_trader, bid_trader = ForestTrader(h=h), ForestTrader(h=h)
    ask_trader.load(model_name='Huorn askclose ' + f, fast=True)
    bid_trader.load(model_name='Huorn bidclose ' + f, fast=True)
    balance_list, profit_list, order_list = {}, {}, {}

    print('{} : S\t initialization took {}'.format(dt.now(), dt.now() - t1))

    while count < 4:

        if dt.now().second == 0 and dt.now().minute % freq == 0:

            t1 = dt.now()
            print('{} : {}\t starting iteration'.format(dt.now(), count))

            # STEP 1: CLOSE OPEN POSITION
            # con.close_all_for_symbol('EUR/USD')
            balance = get_balance(con)
            profit = balance - old_balance
            balance_list[t1] = balance
            profit_list[t1] = profit
            amount = int(10000 * 3 / 100)  # change '10000' for 'balance'
            print('{} : {}\t new balance is {}, profit is {}'.format(dt.now(), count, balance, profit))

            # STEP 2: GET MOST RECENT DATA
            time.sleep(30)  # wait to get last minute data
            df, labels, price = get_price_data(con)
            print('{} : {}\t last collected data is from {}'.format(dt.now(), count, df.index[-1]))

            # STEP 3: PREDICT AND OPEN NEW POSITION
            now_ask, now_bid = get_current_askbid(con)
            print('{} : {}\t last ASK is {} and last BID is {}'.format(dt.now(), count, round(now_ask, 5), round(now_bid, 5)))
            pred_ask = ask_trader.predict_last(df, labels, price)
            pred_bid = bid_trader.predict_last(df, labels, price)

            order = decide_order(amount, now_bid, now_ask, pred_bid, pred_ask)
            order_list[t1] = order
            print('{} : {}\t pred ASK is {} and pred BID is {}, next order is_buy is {}'.format(dt.now(), count, round(pred_ask, 5), round(pred_bid, 5), order['is_buy']))

            # trade(con, order, amount)
            print('{} : {}\t end of iteration, which took {}'.format(dt.now(), count, dt.now() - t1))
            print('-' * 100)
            old_balance = balance
            count += 1

        time.sleep(0.1)

    time.sleep(60 * freq)
    con.close_all_for_symbol('EUR/USD')
    print('{} : E\t trading has been ended'.format(dt.now()))
    return balance_list, profit_list, order_list


if __name__ == "__main__":
    # fetch_currency_rate('./data/dataset_eurgbp.csv', 'EUR', 'GBP', 5, alpha_key)
    # fetch_data()
    # train_models()
    # backtest_models()
    # mega_backtest()

    res = heart_beat()
    print(res)
