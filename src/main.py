import time
import configparser
import pandas as pd
from datetime import datetime as dt

from traders import LstmContextTrader
from api_emulator import Emulator
from utils.basics import write_data, safe_try, omega2assets
from utils.logging import get_logger
from utils.data_fetching import fetch_yahoo_data, fetch_poloniex_data, fetch_intrinio_data

from data_preparation import load_data
from utils.constants import COMPANIES, PERFORMERS, LEVERAGES

logger = get_logger()
config = configparser.ConfigParser()
config.read('../resources/trading212.cfg')
user_name = config['TRADING212']['user_name']
pwd = config['TRADING212']['password']

# Setting constant values
TARGET_COL = 'close'
TRADEFREQ = 1
INITIAL_GAMBLE = 1000
VERSION = 4
H = 10
EPOCHS = 5400
PATIENCE = 300
T0 = '2010-01-01'
T1 = '2019-01-01'
T2 = '2021-01-01'


def train_model(plot=True):
    """
    Trains a model.

    :param bool plot: whether to plot model summary, if appropriate.
    :rtype: None
    """
    print('Training model...')
    folder = '../data/yahoo/'
    # trader = LstmContextTrader(h=H, normalize=True, t0=T0, t1=T1, t2=T2)
    trader = LstmContextTrader(load_from=f'Huorn_v{VERSION}', fast_load=False)
    # df, labels = load_data(folder, T0, T1)
    # trader.ingest_data(df, labels, duplicate=False)
    # trader.save(model_name=f'Huorn_v{VERSION}')
    # trader.train(epochs=EPOCHS, patience=PATIENCE)
    # trader.save(model_name=f'Huorn_v{VERSION}')
    trader.test(test_on='val', plot=plot)


def yesterday_perf():
    print("Correctness of yesterday's predictions")

    folder = '../data/intrinio/'
    df, _ = load_data(folder, T0, T1)
    reco = pd.read_csv('../outputs/recommendations.csv', encoding='utf-8', index_col=0)
    prices = pd.read_csv('../outputs/trade_data.csv', encoding='utf-8', index_col=0)

    date, traded_assets = reco.iloc[-1].name, reco.columns
    exp_accuracy, exp_profits, true_accuracy, true_profits = [], [], [], []
    df, prices, reco = df.loc[date], prices.loc[date], reco.iloc[-2]

    print(f"Computed with data from {reco.name}, traded on {date}.")
    print('_' * 100, '\n')
    print('Asset | Quantity | Reco | Order | Exp P/L | True P/L | Exp Open | True Open | Exp Close | True Close')
    print('-' * 100)
    for i, asset in enumerate(COMPANIES):
        if asset in traded_assets:
            df_row, prices_row = df[df['asset'] == i], prices[prices['asset'] == asset]
            exp_open, exp_close = df_row['open'].values[0], df_row['close'].values[0]
            true_open, true_close = prices_row['open'].values[0], prices_row['close'].values[0]
            true_pl, order = prices_row['result'].values[0], prices_row['is_buy'].values[0]
            asset_reco = reco[asset]
            quantity = int(INITIAL_GAMBLE * LEVERAGES[asset] / exp_open)
            exp_pl = (2 * asset_reco - 1) * (exp_close - exp_open) * quantity
            exp_accuracy.append(asset_reco == (exp_close > exp_open)), exp_profits.append(exp_pl)
            true_accuracy.append(asset_reco == (true_close > true_open)), true_profits.append(true_pl)
            print(f'{asset:5s} | {quantity:8d} | {asset_reco:4d} | {order:5d} | {exp_pl:7.2f} | {true_pl:8.2f} |'
                  f'{exp_open:8.2f} | {true_open:9.2f} | {exp_close:9.2f} | {true_close:10.2f}')
    print('_' * 100, '\n')
    print(f'Expected accuracy was {100 * pd.Series(exp_accuracy).mean():.2f}%.'
          f'True accuracy was {100 * pd.Series(true_accuracy).mean():.2f}%')
    print(f'Expected P/L was {sum(exp_profits):.2f}. True P/L was {sum(true_profits):.2f}.')


def get_recommendations():
    import numpy as np
    now = dt.now()
    folder = '../data/yahoo/'
    trader = LstmContextTrader(load_from=f'Huorn_v{VERSION}', fast_load=True)
    df, labels = load_data(folder, T0, T1, keep_last=True)
    X, P, _, ind = trader.transform_data(df, labels)
    omega = trader.predict(X, P)[-1]
    open_price = np.concatenate(([1.0], P[-1][:, 2]))
    portfolio = omega2assets(INITIAL_GAMBLE, omega, open_price)
    order_book = []
    for i, quantity in enumerate(portfolio[1:]):
        if quantity > 0:
            order = {'asset': PERFORMERS[i], 'is_buy': 1, 'quantity': int(quantity),
                     'data_date': ind[-1], 'current_date': now.strftime("%Y-%m-%d %H:%M:%S")}
            order_book.append(order)
    logger.info(f'recommendations inference took {round((dt.now() - now).total_seconds())} seconds')
    return order_book


def place_orders(order_book):
    emulator = Emulator(user_name, pwd)
    emulator.close_all_trades()
    for order in order_book:
        emulator.open_trade(order)
    prices = emulator.get_current_prices()
    prices = pd.DataFrame([prices]).set_index('date', drop=True)
    path = '../outputs/open_prices.csv'
    write_data(path, prices)
    emulator.quit()


def close_orders():
    emulator = Emulator(user_name, pwd)
    emulator.close_all_trades()
    prices = emulator.get_trades_results()
    prices = pd.DataFrame(prices).set_index('date', drop=True)
    path = '../outputs/trade_data.csv'
    write_data(path, prices, same_ids=True)
    emulator.quit()


def get_trades_results():
    emulator = Emulator(user_name, pwd)
    prices = emulator.get_trades_results()
    prices = pd.DataFrame(prices).set_index('date', drop=True)
    path = '../outputs/trade_data.csv'
    write_data(path, prices, same_ids=True)
    emulator.quit()


def heartbeat():
    """
    Live trading program, running forever.
    :rtype: None
    """

    logger.info('launching heartbeat')
    order_book = []
    while True:
        now = dt.now()
        if now.minute % 10 == 0 and now.second == 0:
            logger.info('still running')
        if now.hour == 15 and now.minute == 15 and now.second == 0:
            logger.info('updating data')
            safe_try(fetch_yahoo_data)
            logger.info('updating was successful')
        if now.hour == 15 and now.minute == 25 and now.second == 0:
            logger.info('computing orders')
            order_book = safe_try(get_recommendations)
            logger.info('computing was successful')
        if now.hour == 15 and now.minute == 29 and now.second == 45:
            logger.info('placing orders')
            safe_try(place_orders, order_book)
            logger.info('placing was successful')
        if now.hour == 21 and now.minute == 50 and now.second == 0:
            logger.info('closing all orders')
            safe_try(close_orders)
            logger.info('closing was successful')
        time.sleep(1)


if __name__ == "__main__":

    # fetch_yahoo_data(companies=COMPANIES)
    train_model()

    # o = get_recommendations()
    # print(o)
    # place_orders(o)
    # get_trades_results()
    # yesterday_perf()
    # heartbeat()

    # for pair in PAIRS:
    #     folder = '../data/poloniex/'
    #     path = folder + pair.lower()
    #     df = pd.read_csv(f'{path}.csv', index_col=0)
    #     df = df[df.index > '2000']
    #     print(pair, df.index.min())
