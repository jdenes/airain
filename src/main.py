import time
import configparser
import numpy as np
import pandas as pd
from datetime import datetime as dt

from traders import LstmContextTrader
from api_emulator import Emulator
from utils.basics import write_data, safe_try, omega2assets
from utils.logging import get_logger
from utils.data_fetching import fetch_yahoo_data

from data_preparation import load_data
from utils.constants import DJIA, DJIA_PERFORMERS, LEVERAGES

logger = get_logger()
config = configparser.ConfigParser()
config.read('../resources/trading212.cfg')
user_name = config['TRADING212']['user_name']
pwd = config['TRADING212']['password']

# Setting constant values
TARGET_COL = 'close'
FOLDER = '../data/yahoo/'
COMPANIES = DJIA_PERFORMERS
TRADEFREQ = 1
INITIAL_GAMBLE = 45000
VERSION = 'test'
H = 10
EPOCHS = 5000  # 1700
PATIENCE = 500
T0 = '2010-01-01'
T1 = '2019-01-01'
T2 = '2020-01-01'


def train_model(plot=True):
    """
    Trains a model.

    :param bool plot: whether to plot model summary, if appropriate.
    :rtype: None
    """
    print('Training model...')
    trader = LstmContextTrader(h=H, normalize=True, t0=T0, t1=T1, t2=T2,
                               noise_level=3.0, layer_coefficient=1.0,
                               entropy_lambda=0e-4, learning_rate=1e-6)
    # trader = LstmContextTrader(load_from=f'ContextTrader_{VERSION}', fast_load=False)
    df, labels = load_data(FOLDER, COMPANIES, T0, T1)
    trader.ingest_data(df, labels, duplicate=False)
    trader.train(epochs=EPOCHS, patience=PATIENCE, verbose=100)
    trader.save(model_name=f'ContextTrader_{VERSION}')
    trader.test(companies=COMPANIES, test_on='test', plot=plot, noise=False)
    # trader.test(companies=COMPANIES, test_on='train', plot=plot, noise=False)


def grid_search():
    grid = []
    df, labels = load_data(FOLDER, COMPANIES, T0, T1)
    for h in [10]:
        for coeff in [1.0]:
            for noise in [0.5, 1.0, 2.0, 3.0, 5.0]:
                for ent in [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]:
                    logger.info(f"H: {h} - Coeff: {coeff:.1f} - Noise: {noise:.3f} - Entropy: {ent:.4f}")
                    trader = LstmContextTrader(h=h, normalize=True, t0=T0, t1=T1, t2=T2,
                                               noise_level=noise, layer_coefficient=coeff,
                                               entropy_lambda=ent, learning_rate=1e-4)
                    trader.ingest_data(df, labels, duplicate=False, verbose=0)
                    trader.train(epochs=50000, patience=1000, verbose=0)
                    balance = trader.test(companies=COMPANIES, test_on='test', verbose=0, plot=False)
                    grid.append({'h': h, 'coeff': coeff, 'ent': ent, 'noise': noise, 'balance': balance})
                    logger.info(f"H: {h} - Coeff: {coeff:.1f} - Noise: {noise:.3f} - Entropy: {ent:.4f} - "
                                f"Balance: {balance:.2f}")
    logger.info(grid)
    logger.info(pd.DataFrame(grid))
    pd.DataFrame(grid).to_csv('../outputs/gridsearch_2.csv')


def yesterday_perf():
    print("Correctness of yesterday's predictions")

    df, _ = load_data(FOLDER, COMPANIES, T0, T1)
    reco = pd.read_csv('../outputs/recommendations.csv', encoding='utf-8', index_col=0)
    prices = pd.read_csv('../outputs/trade_data.csv', encoding='utf-8', index_col=0)

    date, traded_assets = reco.iloc[-1].name, reco.columns
    exp_accuracy, exp_profits, true_accuracy, true_profits = [], [], [], []
    df, prices, reco = df.loc[date], prices.loc[date], reco.iloc[-2]

    print(f"Computed with data from {reco.name}, traded on {date}.")
    print('_' * 100, '\n')
    print('Asset | Quantity | Reco | Order | Exp P/L | True P/L | Exp Open | True Open | Exp Close | True Close')
    print('-' * 100)
    for i, asset in enumerate(DJIA):
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
    now = dt.now()
    trader = LstmContextTrader(load_from=f'ContextTrader_{VERSION}', fast_load=True)
    df, labels = load_data(FOLDER, COMPANIES, T0, T1, keep_last=True)
    bound = sorted(df.index.unique())[-H]
    df, labels = df[df.index >= bound], labels[df.index >= bound]
    X, P, _, ind = trader.transform_data(df, labels)
    omega = trader.predict(X, P)[-1]
    open_price = np.concatenate(([1.0], P[-1][:, 2]))
    portfolio = omega2assets(INITIAL_GAMBLE, omega, open_price)
    order_book = []
    for company, quantity in zip(COMPANIES, portfolio[1:]):
        if quantity > 0:
            order = {'asset': company, 'is_buy': 1, 'quantity': int(quantity),
                     'data_date': ind[-1], 'current_date': now.strftime("%Y-%m-%d %H:%M:%S")}
            order_book.append(order)
    logger.info(f'recommendations inference took {round((dt.now() - now).total_seconds())} seconds')
    return order_book


def place_orders(order_book):
    emulator = Emulator(user_name, pwd).start()
    emulator.close_all_trades()
    for order in order_book:
        emulator.open_trade(order)
    emulator.quit()


def get_trades_results():
    emulator = Emulator(user_name, pwd).start()
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
        if now.hour == 21 and now.minute == 57 and now.second == 30:  # 17sec
            logger.info('updating data')
            safe_try(fetch_yahoo_data, DJIA)
            logger.info('updating was successful')
            logger.info('computing orders')
            order_book = safe_try(get_recommendations)
            logger.info('computing was successful')
        if now.hour == 21 and now.minute == 58 and now.second == 30:  # 30sec
            logger.info('placing orders')
            safe_try(place_orders, order_book)
            logger.info('placing was successful')
        time.sleep(1)


if __name__ == "__main__":

    # fetch_yahoo_data(companies=DAX)
    # fetch_yahoo_data(companies=CAC40)
    fetch_yahoo_data(companies=DJIA)
    # fetch_poloniex_data(pairs=PAIRS)
    train_model()
    # grid_search()
    # heartbeat()
