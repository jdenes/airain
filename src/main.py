import time
import logging
import configparser
import pandas as pd
from datetime import datetime as dt

from traders import LstmTrader
from api_emulator import Emulator
from utils import fetch_intrinio_news, fetch_intrinio_prices, write_data
from utils import load_data, nice_plot

# Configuring setup constants
# noinspection PyArgumentList
logging.basicConfig(handlers=[logging.FileHandler("../logs/LOG.log"), logging.StreamHandler()],
                    format='%(asctime)s: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

config = configparser.ConfigParser()
config.read('../resources/intrinio.cfg')
api_key = config['INTRINIO']['access_token']
config.read('../resources/trading212.cfg')
user_name = config['TRADING212']['user_name']
pwd = config['TRADING212']['password']

# Setting constant values
VERSION = 2
UNIT = 'H'  # 'm' or 'd'
DATAFREQ = 1
TRADEFREQ = 1
H = 30
INITIAL_GAMBLE = 2000
EPOCHS = 50
TARGET_COL = 'close'
CURR = 'EUR/USD'
LOWER_CURR = 'eurusd'

# Removed UTX
companies = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'MSFT', 'IBM', 'CVX', 'JNJ', 'PG', 'PFE', 'VZ', 'BA', 'MRK',
             'CSCO', 'HD', 'MCD', 'MMM', 'GE', 'NKE', 'CAT', 'V', 'JPM', 'AXP', 'GS', 'UNH', 'TRV']
leverages = {'AAPL': 5, 'XOM': 5, 'KO': 5, 'INTC': 5, 'WMT': 5, 'MSFT': 5, 'IBM': 5, 'CVX': 5, 'JNJ': 5,
             'PG': 5, 'PFE': 5, 'VZ': 5, 'BA': 5, 'MRK': 5, 'CSCO': 5, 'HD': 5, 'MCD': 5, 'MMM': 5,
             'GE': 5, 'NKE': 5, 'CAT': 5, 'V': 5, 'JPM': 5, 'AXP': 5, 'GS': 5, 'UNH': 5, 'TRV': 5}
# performers = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'IBM', 'CVX', 'JNJ', 'PG', 'VZ', 'MRK', 'HD', 'GE', 'GS']
performers = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'MSFT', 'CVX', 'MMM', 'V', 'GS']


def fetch_intrinio_data():
    for company in companies:
        print(f'Fetching {company} data...')
        path = f'../data/intrinio/{company.lower()}'
        fetch_intrinio_news(filename=path + '_news.csv', api_key=api_key, company=company)
        fetch_intrinio_prices(filename=path + '_prices.csv', api_key=api_key, company=company)


def train_model():
    print('Training model...')
    trader = LstmTrader(h=H, normalize=False)
    # banks = [f[:-4] for f in os.listdir('../data/finance/') if f.endswith('.csv')]
    # banks = companies

    df, labels = load_data('../data/intrinio/', TRADEFREQ, DATAFREQ)
    trader.ingest_traindata(df, labels)

    trader.train(epochs=EPOCHS)
    trader.test(plot=False)
    trader.save(model_name=f'Huorn_v{VERSION}')


def backtest(plot=False):
    print('_' * 100, '\n')
    print('Initializing backtest...')
    trader = LstmTrader(load_from=f'Huorn_v{VERSION}')
    ov_df, ov_labels = load_data('../data/intrinio/', TRADEFREQ, DATAFREQ, start_from=trader.t2)
    assets_profits, assets_returns, assets_balance, assets_pls_hist = [], [], [], []
    index_hist = None

    for asset in enumerate(companies):
        if asset[1] in performers:
            print('_' * 100, '\n')
            print(f'Backtesting on {asset[1]}...\n')

            buy, sell, buy_correct, sell_correct, do_nothing = 0, 0, 0, 0, 0
            index_hist, balance_hist, returns_hist, pls_hist, bench_hist = [], [], [], [], []
            balance = bench_balance = INITIAL_GAMBLE

            df, labels = ov_df[ov_df['asset'] == asset[0]], ov_labels[ov_df['asset'] == asset[0]]
            X, P, y, ind = trader.transform_data(df, labels, get_index=True)
            df = df.loc[ind]

            preds = trader.predict(X, P)
            # y_true, y_pred = pd.Series(y.flatten()), pd.Series(preds)
            # from sklearn.metrics import classification_report
            # print(classification_report(y_true, y_pred, digits=4))

            for i in range(1, len(df)):

                j, k = ind[i], ind[i - 1]
                if dt.strptime(j, '%Y-%m-%d').minute % TRADEFREQ == 0:

                    open_price, close_price = df.loc[j]['open'], df.loc[j]['close']
                    pl, gpl = 0, 0
                    # quantity = int(balance * 3 / 100)
                    amount = INITIAL_GAMBLE
                    quantity = int(amount * leverages[asset[1]] / open_price)

                    # Step 1 (morning) : decide what to do today and open position
                    pred, label = preds[i - 1], labels[ind[i - 1]]
                    order = decide_order(asset[1], quantity, open_price, pred, j)

                    # Step 2 (evening): close position
                    if order['is_buy'] is not None:
                        if order['is_buy']:
                            pl = round((close_price - order['open']) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)
                            buy_correct += int(pl >= 0)
                            buy += 1
                        else:
                            pl = round((order['open'] - close_price) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)
                            sell_correct += int(pl >= 0)
                            sell += 1
                    else:
                        do_nothing += 1

                    # Step 3 bis: compute metrics and stuff
                    balance, returns, pls = round(balance + pl, 2), round((pl / amount) * 100, 2), round(pl, 2)
                    bench_balance = round(bench_balance + (close_price - order['open']) * quantity, 2)
                    balance_hist.append(balance), returns_hist.append(returns), pls_hist.append(pls)
                    bench_hist.append(bench_balance)
                    index_hist.append(ind[i])

            pos_pls, neg_pls = [i for i in pls_hist if i > 0], [i for i in pls_hist if i < 0]
            profit = round(balance - INITIAL_GAMBLE, 2)
            bench = round(bench_balance - INITIAL_GAMBLE, 2)
            buy_acc = round(100 * buy_correct / buy, 2) if buy != 0 else 'NA'
            sell_acc = round(100 * sell_correct / sell, 2) if sell != 0 else 'NA'
            pos_days = round(100 * sum([i >= 0 for i in pls_hist]) / len(pls_hist), 2)
            m_returns = round(sum(returns_hist) / len(returns_hist), 2)
            m_wins = round(sum(pos_pls) / len(pos_pls), 2) if len(pos_pls) != 0 else 'NA'
            m_loss = round(sum(neg_pls) / len(neg_pls), 2) if len(neg_pls) != 0 else 'NA'

            assets_profits.append(profit), assets_returns.append(m_returns)
            assets_balance.append(balance_hist), assets_pls_hist.append(pls_hist)
            print(f'Profit: {profit}. Benchmark: {bench}. Mean daily return: {m_returns}%.')
            print(f'Correct moves: {pos_days}%. Correct buy: {buy_acc}%. Correct sell: {sell_acc}%.')
            print(f'Av. wins/loss amounts: {m_wins}/{m_loss}. Ext. wins/loss amounts: {max(pls_hist)}/{min(pls_hist)}.')
            if plot:
                nice_plot(index_hist, [balance_hist, bench_hist], ['Algorithm', 'Benchmark'],
                          title=f'Profit evolution for {asset[1]}')

    print('_' * 100, '\n')
    if plot:
        portfolio_balance_hist = [sum([pls[i] for pls in assets_balance]) for i in range(len(assets_pls_hist[0]))]
        nice_plot(index_hist, [portfolio_balance_hist], ['Portfolio balance'], title=f'Portfolio balance evolution')

    portfolio_pls_hist = [sum([pls[i] for pls in assets_pls_hist]) for i in range(len(assets_pls_hist[0]))]
    portfolio_mean_pls = round(sum(portfolio_pls_hist) / len(portfolio_pls_hist), 2)

    n_pos = len([x for x in assets_profits if x > 0])
    m_prof = round(sum(assets_profits) / len(assets_profits), 2)
    m_ret = round(sum(assets_returns) / len(assets_returns), 2)
    print('Returns:', assets_returns)
    print(f'Average profit by assets: {m_prof}. Average daily return: {m_ret}%.')
    print(f'Profitable assets: {n_pos}/{len(assets_profits)}.')
    print(f'Average daily profit of portfolio: {portfolio_mean_pls}.')
    print('_' * 100, '\n')
    # perf = [(companies[i], ret) for i, ret in enumerate(assets_returns) if ret in sorted(assets_returns)[::-1][:11]]
    # perf = [companies[i] for i, ret in enumerate(assets_returns) if ret in sorted(assets_returns)[::-1][:11]]
    # print(perf)


def decide_order(asset, quantity, open_price, pred, date):
    # if pred_bid > (1 - tolerance) * now_ask:
    if pred == 1:
        order = {'asset': asset, 'is_buy': True, 'open': open_price, 'quantity': quantity, 'date': date}
    # elif pred_ask < now_bid / (1 - tolerance):
    elif pred == 0:
        order = {'asset': asset, 'is_buy': False, 'open': open_price, 'quantity': quantity, 'date': date}
    else:
        order = {'asset': asset, 'is_buy': None, 'open': open_price, 'quantity': quantity, 'date': date}
    return order


def get_yesterday_perf():
    print("Correctness of yesterday's predictions")

    df, _ = load_data('../data/intrinio/', TRADEFREQ, DATAFREQ)
    reco = pd.read_csv('../outputs/recommendations.csv', encoding='utf-8', index_col=0)
    prices = pd.read_csv('../outputs/trade_data.csv', encoding='utf-8', index_col=0)

    date, traded_assets = reco.iloc[-1].name, reco.columns
    exp_accuracy, exp_profits, true_accuracy, true_profits = [], [], [], []
    df, prices, reco = df.loc[date], prices.loc[date], reco.iloc[-2]

    print(f"Computed with data from {reco.name}, traded on {date}.")
    print('_' * 100, '\n')
    print('Asset | Quantity | Reco | Order | Exp P/L | True P/L | Exp Open | True Open | Exp Close | True Close')
    print('-' * 100)
    for i, asset in enumerate(companies):
        if asset in traded_assets:
            df_row, prices_row = df[df['asset'] == i], prices[prices['asset'] == asset]
            exp_open, exp_close = df_row['open'].values[0], df_row['close'].values[0]
            true_open, true_close = prices_row['open'].values[0], prices_row['close'].values[0]
            true_pl, order = prices_row['result'].values[0], prices_row['is_buy'].values[0]
            asset_reco = reco[asset]
            quantity = int(INITIAL_GAMBLE * leverages[asset] / exp_open)
            exp_pl = (2 * asset_reco - 1) * (exp_close - exp_open) * quantity
            exp_accuracy.append(asset_reco == (exp_close > exp_open)), exp_profits.append(exp_pl)
            true_accuracy.append(asset_reco == (true_close > true_open)), true_profits.append(true_pl)
            print('{:5s} | {:8d} | {:4d} | {:5d} | {:7.2f} | {:8.2f} | {:8.2f} | {:9.2f} | {:9.2f} | {:10.2f}'.format(
                   asset, quantity, asset_reco, order, exp_pl, true_pl, exp_open, true_open, exp_close, true_close))
    print('_' * 100, '\n')
    print('Expected accuracy was {:.2f}%. True accuracy was {:.2f}%'.format(100 * pd.Series(exp_accuracy).mean(),
                                                                            100 * pd.Series(true_accuracy).mean()))
    print('Expected P/L was {:.2f}. True P/L was {:.2f}.'.format(pd.Series(exp_profits).sum(),
                                                                 pd.Series(true_profits).sum()))


def update_data():
    folder = '../data/intrinio/'
    for company in companies:
        path = folder + company.lower()
        fetch_intrinio_news(filename=path + '_news.csv', api_key=api_key, company=company, update=True)
        fetch_intrinio_prices(filename=path + '_prices.csv', api_key=api_key, company=company, update=True)
    load_data(folder, TRADEFREQ, DATAFREQ, update_embed=True)


def get_recommendations():
    now = time.time()
    trader = LstmTrader(load_from=f'Huorn_v{VERSION}')
    df, labels = load_data('../data/intrinio/', TRADEFREQ, DATAFREQ)
    yesterday = df.index.max()
    df = df.loc[yesterday].reset_index(drop=True)
    X, P, _, ind = trader.transform_data(df, labels, get_index=True)
    preds = trader.predict(X, P)
    reco, order_book = {'date': yesterday}, []
    lev = pd.Series([leverages[co] for co in companies])
    quantity = (INITIAL_GAMBLE * (lev / df['close'])).astype(int)
    for i, pred in enumerate(preds):
        if companies[i] in performers:
            reco[companies[i]] = pred
            order_book.append({'asset': companies[i], 'is_buy': pred, 'quantity': int(quantity[i])})
    path = '../outputs/recommendations.csv'
    reco = pd.DataFrame([reco]).set_index('date', drop=True)
    write_data(path, reco)
    logger.info(f'recommendations inference took {round(time.time() - now)} seconds')
    return order_book


def place_orders(order_book):
    emulator = Emulator(user_name, pwd)
    emulator.close_all_trades()
    for order in order_book:
        emulator.open_trade(order)
    prices = emulator.get_open_prices()
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


def safe_try(function, arg=None, max_attempts=1000):
    res, attempts = None, 0
    while attempts < max_attempts:
        try:
            if arg is None:
                res = function()
            else:
                res = function(arg)
        except Exception as ex:
            attempts += 1
            logger.warning(f"execution of function {function.__name__} failed {attempts} times. Exception met: {ex}")
            continue
        break
    if attempts == max_attempts:
        logger.error(f"too many attempts to safely execute function {function.__name__}")
        raise Exception("Too many attempts to safely execute")
    return res


def heartbeat():
    logger.info('launching heartbeat')
    while True:
        now = dt.now()
        if now.minute % 10 == 0 and now.second == 0:
            logger.info('still running')
        if now.hour == 14 and now.minute == 30 and now.second == 0:
            logger.info('updating data')
            safe_try(update_data)
            logger.info('updating was successful')
        if now.hour == 15 and now.minute == 29 and now.second == 30:
            logger.info('placing orders')
            order_book = safe_try(get_recommendations)
            safe_try(place_orders, order_book)
            logger.info('placing was successful')
        if now.hour == 21 and now.minute == 50 and now.second == 0:
            logger.info('closing all orders')
            safe_try(close_orders)
            logger.info('closing was successful')
        time.sleep(1)


if __name__ == "__main__":
    # fetch_intrinio_data()
    # update_data()
    # train_model()
    backtest(plot=False)
    # get_recommendations()
    # get_trades_results()
    # get_yesterday_perf()
    # heartbeat()
