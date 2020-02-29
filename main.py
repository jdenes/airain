import time
import fxcmpy
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt

from traders import ForestTrader, Dummy, Randommy, IdealTrader, LstmTrader
from utils import load_data, fetch_crypto_rate, fetch_currency_rate, fetch_fxcm_data, nice_plot

datafreq = 1
tradefreq = 1
f, tf = str(datafreq), str(tradefreq)
lag = 1
h = 10
initial_gamble = 10000
fees = 0.0
tolerance = 0e-5  # 2
shift = tradefreq + lag
enrich = True

account_id = '1195258'
alpha_key = "H2T4H92C43D9DT3D"
fxcm_key = '9c9f8a5725072aa250c8bd222dee004186ffb9e0'

cols = 'date,pred ask,true ask,pred ask diff,true ask diff,pred bid,true bid,pred bid diff,true bid diff,tbuy,pbuy,' \
       'tsell,psell\n '


def fetch_data():
    con = fxcmpy.fxcmpy(access_token=fxcm_key, server='demo')
    start, end = '2017-01-01 00:00:00', '2020-01-30 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_eurusd_train_' + f + '.csv', start=start, end=end, freq=datafreq, con=con)
    start, end = '2020-02-01 00:00:00', '2020-02-28 00:00:00'
    fetch_fxcm_data(filename='./data/dataset_eurusd_test_' + f + '.csv', start=start, end=end, freq=datafreq, con=con)


def train_models():
    print('Training ASK model...')
    df, labels, price = load_data('dataset_eurusd_train', target_col='askopen', shift=shift, datafreq=datafreq)
    trader = LstmTrader(h=h)
    trader.ingest_traindata(df=df, labels=labels)
    trader.train()  # n_estimators=100)
    print(trader.test(plot=True))
    trader.save(model_name='Huorn askopen NOW' + tf)
    del trader, df, labels, price
    print('Training BID model...')
    df, labels, price = load_data('dataset_eurusd_train', target_col='bidopen', shift=shift, datafreq=datafreq)
    print(df)
    print(labels)
    trader = LstmTrader(h=h)
    trader.ingest_traindata(df=df, labels=labels)
    trader.train()  # n_estimators=100)
    print(trader.test(plot=True))
    trader.save(model_name='Huorn bidopen NOW' + tf)
    del trader, df, labels, price


def backtest_models():
    curves, names = [], []
    df, labels, price = load_data(filename='dataset_eurusd_test', target_col='askopen', shift=shift, datafreq=datafreq)
    ask_trader = LstmTrader(h=h)
    ask_trader.load(model_name='Huorn askopen NOW' + tf, fast=True)
    ask_backtest = ask_trader.backtest(df, labels, price, tradefreq, lag, initial_gamble, fees)
    curves.append(ask_backtest['value']), names.append('ASK model')
    del ask_trader

    df, labels, price = load_data(filename='dataset_eurusd_test', target_col='bidopen', shift=shift, datafreq=datafreq)
    bid_trader = LstmTrader(h=h)
    bid_trader.load(model_name='Huorn bidopen NOW' + tf, fast=True)
    bid_backtest = bid_trader.backtest(df, labels, price, tradefreq, lag, initial_gamble, fees)
    curves.append(bid_backtest['value']), names.append('BID model')
    del bid_trader

    baseline = Dummy().backtest(df, labels, price, tradefreq, lag, initial_gamble, fees)
    curves.append(baseline['value'][:-int(enrich)]), names.append('Pure USD')
    random = Randommy().backtest(df, labels, price, tradefreq, lag, initial_gamble, fees)
    curves.append(random['value'][:-int(enrich)]), names.append('Random')

    nice_plot(ind=ask_backtest['index'], curves_list=curves, names=names,
              title='Equity evolution, without spread, tradefreq ' + str(tradefreq))


def mega_backtest():
    df, labels, _ = load_data('dataset_eurusd_test', 'askopen', shift, datafreq, keep_last=True)
    ask_trader, bid_trader = LstmTrader(h=h), LstmTrader(h=h)
    ask_trader.load(model_name='Huorn askopen NOW' + tf, fast=True)
    bid_trader.load(model_name='Huorn bidopen NOW' + tf, fast=True)
    # print(max([estimator.tree_.max_depth for estimator in bid_trader.model.estimators_]))

    buy, sell, buy_correct, sell_correct, do_nothing = 0, 0, 0, 0, 0
    index_list, profit_list = [], []
    order = {'is_buy': None}

    balance = initial_gamble

    X, _, ind = ask_trader.transform_data(df, labels, get_index=True)
    df = df.loc[ind]

    ask_preds = ask_trader.predict(X)
    bid_preds = bid_trader.predict(X)
    # ask_preds = df['askopen'].shift(-shift).to_list()
    # bid_preds = df['bidopen'].shift(-shift).to_list()

    with open('./resources/report_backtest.csv', 'w') as file:
        file.write(cols)

    for i in range(lag, len(df)):

        j = ind[i]
        if dt.strptime(j, '%Y-%m-%d %H:%M:%S').minute % tradefreq == 0:

            now_ask, now_bid = df.loc[j]['askopen'], df.loc[j]['bidopen']
            gpl = 0
            amount = int(balance * 3 / 100)

            # Step 0: stats
            if i > lag:
                info = [j, pred_ask, now_ask, 1000 * (pred_ask - old_ask), 1000 * (now_ask - old_ask),
                        pred_bid, now_bid, 1000 * (pred_bid - old_bid), 1000 * (now_bid - old_bid),
                        now_bid > old_ask, pred_bid > old_ask, now_ask < old_bid, pred_ask < old_bid]
                with open('./resources/report_backtest.csv', 'a') as file:
                    file.write(','.join([str(x) for x in info]) + '\n')

            # Step two: close former position is needed, else continue
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

            # Step one: decide what to do next
            pred_ask, pred_bid = ask_preds[i - lag], bid_preds[i - lag]
            order = decide_order(amount, now_bid, now_ask, pred_bid, pred_ask)

            balance = round(balance + gpl, 2)
            profit_list.append(balance)
            index_list.append(ind[i])
            old_ask, old_bid = now_ask, now_bid

    no_trade = round(100 * do_nothing / len(index_list), 3)
    buy_acc = round(100 * buy_correct / buy, 3)
    sell_acc = round(100 * sell_correct / sell, 3)

    print('Overall profit: {} | Correct share buy {} | Correct share sell {} | Share of done nothing {}'.format(
        round(balance - initial_gamble, 2), buy_acc, sell_acc, no_trade))
    nice_plot(index_list, [profit_list], ['Profit evolution'],
              title='Equity evolution, with spread, tradefreq ' + str(tradefreq))


def gross_pl(pl, K, price):
    return round((K / 10) * pl * (1 / price), 2)


def get_price_data(con):
    fetch_fxcm_data('./data/dataset_eurusd_now_' + f + '.csv', freq=datafreq, con=con, n_last=30)
    df, labels, price = load_data(filename='dataset_eurusd_now', target_col='askopen',
                                  shift=shift,
                                  datafreq=datafreq,
                                  keep_last=True)
    return df, labels, price


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
    ask_trader.load(model_name='Huorn askopen NOW' + tf, fast=True)
    bid_trader.load(model_name='Huorn bidopen NOW' + tf, fast=True)
    balance_list, profit_list, order_list = {}, {}, {}

    order = {'is_buy': None}
    buy, sell, buy_correct, sell_correct, do_nothing = 0, 0, 0, 0, 0

    with open('./resources/report.csv', 'w') as file:
        file.write(cols)

    print('{} : S\t initialization took {}'.format(dt.now(), dt.now() - t1))

    while True:

        if dt.now().second == 0 and dt.now().minute % 1 == 0:

            time.sleep(0)
            t1 = dt.now()
            print('{} : {}\t starting iteration'.format(dt.now(), count))
            now_ask, now_bid = get_current_askbid(con)

            # STEP 1: CLOSE OPEN POSITION
            # con.close_all_for_symbol('EUR/USD')
            balance = get_balance(con)
            profit = round(balance - old_balance, 2)
            balance_list[t1] = balance
            profit_list[t1] = profit
            amount = int(1000 * 30 / 1000)  # change '1000' for 'balance'
            print('{} : {}\t new balance is {}, profit is {}'.format(dt.now(), count, balance, profit))

            if count > 1 and order['is_buy'] is not None:
                if order['is_buy']:
                    buy += 1
                    buy_correct += int(profit > 0)
                else:
                    sell += 1
                    sell_correct += int(profit > 0)
            else:
                do_nothing += 1

            # STEP 2: GET MOST RECENT DATA
            df, labels, price = get_price_data(con)

            print('{} : {}\t last collected data is from {}'.format(dt.now(), count, df.index[-1]))
            new_k = df.loc[df.index[-1]]
            if count > 1:
                print(round(new_k['askopen'], 5), round(old_ask, 5), round(new_k['bidopen'], 5), round(old_bid, 5))
                info = [df.index[-1], pred_ask, now_ask, 1000 * (pred_ask - old_ask), 1000 * (now_ask - old_ask),
                        pred_bid, now_bid, 1000 * (pred_bid - old_bid), 1000 * (now_bid - old_bid),
                        now_bid > old_ask, pred_bid > old_ask, now_ask < old_bid, pred_ask < old_bid]
                with open('./resources/report.csv', 'a') as file:
                    file.write(','.join([str(x) for x in info]) + '\n')

            # STEP 3: PREDICT AND OPEN NEW POSITION
            # now_ask, now_bid = get_current_askbid(con)
            print('{} : {}\t last ASK is {} and last BID is {}'.format(
                dt.now(), count, round(now_ask, 5), round(now_bid, 5)))
            pred_ask = ask_trader.predict_last(df, labels)
            pred_bid = bid_trader.predict_last(df, labels)

            order = decide_order(amount, now_bid, now_ask, pred_bid, pred_ask)
            order_list[t1] = order
            print('{} : {}\t pred ASK is {} and pred BID is {}, next order is_buy is {}'.format(
                dt.now(), count, round(pred_ask, 5), round(pred_bid, 5), order['is_buy']))

            # trade(con, order, amount)
            print('{} : {}\t end of iteration, which took {}'.format(dt.now(), count, dt.now() - t1))

            no_trade = round(100 * do_nothing / count, 3)
            buy_acc = round(100 * buy_correct / buy, 3) if buy > 0 else 'NA'
            sell_acc = round(100 * sell_correct / sell, 3) if sell > 0 else 'NA'
            print('{} : {}\t stats: correct buy: {}%, correct sell: {}%, share non traded: {}%'.format(
                dt.now(), count, buy_acc, sell_acc, no_trade))

            old_balance = balance
            old_ask, old_bid = now_ask, now_bid
            count += 1
            print('-' * 100)

        time.sleep(0.1)

    time.sleep(60 * datafreq)
    con.close_all_for_symbol('EUR/USD')
    print('{} : E\t trading has been ended'.format(dt.now()))
    return balance_list, profit_list, order_list


if __name__ == "__main__":
    # fetch_currency_rate('./data/dataset_eurgbp.csv', 'EUR', 'GBP', 5, alpha_key)
    # fetch_data()
    train_models()
    backtest_models()
    mega_backtest()

    # res = heart_beat()
