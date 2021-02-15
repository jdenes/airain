import time
import configparser
import pandas as pd
from datetime import datetime as dt

from traders import LGBMTrader, LstmContextTrader
from api_emulator import Emulator
from utils.plots import nice_plot
from utils.basics import write_data
from utils.logging import get_logger
from utils.metrics import benchmark_metrics, benchmark_portfolio_metric
from utils.data_fetching import fetch_intrinio_news, fetch_intrinio_prices
from utils.data_fetching import fetch_yahoo_news, fetch_yahoo_prices, fetch_yahoo_intraday

from data_preparation import load_data, precompute_embeddings
from utils.constants import COMPANIES, PERFORMERS, LEVERAGES

logger = get_logger()
config = configparser.ConfigParser()
config.read('../resources/trading212.cfg')
user_name = config['TRADING212']['user_name']
pwd = config['TRADING212']['password']

# Setting constant values
UNIT = 'H'  # 'm' or 'd'
TARGET_COL = 'close'
DATAFREQ = 1
TRADEFREQ = 1
INITIAL_GAMBLE = 10000
VERSION = 3
H = 30
EPOCHS = 50
T0 = '2000-01-01'
T1 = '2019-01-01'
T2 = '2020-01-01'


def fetch_intrinio_data():
    """
    Fetches news and prices data for all companies.

    :rtype: None
    """
    folder = '../data/intrinio/'
    parser = configparser.ConfigParser()
    parser.read('../resources/intrinio.cfg')
    api_key = parser['INTRINIO']['access_token']
    for company in COMPANIES:
        print(f'Fetching {company} data...')
        path = folder + company.lower()
        fetch_intrinio_news(filename=path + '_news.csv', api_key=api_key, company=company)
        fetch_intrinio_prices(filename=path + '_prices.csv', api_key=api_key, company=company)


def fetch_yahoo_data():
    """
    Fetches or updates news and prices data for all companies.

    :rtype: None
    """
    folder = '../data/yahoo/'
    for company in COMPANIES:
        print(f'Fetching {company} data...')
        path = folder + company.lower()
        fetch_yahoo_news(filename=path + '_news.csv', company=company)
        fetch_yahoo_prices(filename=path + '_prices.csv', company=company)
        fetch_yahoo_intraday(f'../data/yahoo_intraday/{company.lower()}_prices.csv', company=company)
    fetch_yahoo_prices(filename=folder + '^n225_prices.csv', company='^n225')
    precompute_embeddings(folder)


def train_model(plot=True):
    """
    Trains a model.

    :param bool plot: whether to plot model summary, if appropriate.
    :rtype: None
    """
    print('Training model...')
    folder = '../data/yahoo/'
    trader = LstmContextTrader(h=H, normalize=False, t0=T0, t1=T1, t2=T2)
    # trader = LstmContextTrader(load_from=f'Huorn_v{VERSION}', fast_load=False)
    df, labels = load_data(folder, T0, T1)
    trader.ingest_traindata(df, labels, duplicate=False)
    trader.save(model_name=f'Huorn_v{VERSION}')
    trader.train(epochs=EPOCHS)
    trader.save(model_name=f'Huorn_v{VERSION}')
    trader.test(plot=plot)


def backtest(plot=False, precomputed_df=None, precomputed_labels=None):
    """
    Backtests a previously trained model.

    :param bool plot: whether to plot profit curves during backtest.
    :param pd.DataFrame precomputed_df: precomputed dataframe.
    :param pd.DataFrame precomputed_labels: precomputed labels.
    :returns: a few portfolio metrics.
    :rtype: dict
    """
    print('_' * 100, '\n')
    print('Initializing backtest...')
    folder = '../data/yahoo/'
    trader = LGBMTrader(load_from=f'Huorn_v{VERSION}')
    if precomputed_df is None or precomputed_labels is None:
        ov_df, ov_labels = load_data(folder, T0, T1, start_from=trader.t2)
    else:
        ov_df, ov_labels = precomputed_df, precomputed_labels
    assets_balance, benchmarks_balance = [], []
    index_hist = None

    for asset in enumerate(COMPANIES):
        if asset[1] in PERFORMERS:
            print('_' * 100, '\n')
            print(f'Backtesting on {asset[1]}...\n')

            index_hist, balance_hist, bench_hist, orders_hist = [], [], [], []
            balance = bench_balance = INITIAL_GAMBLE

            df, labels = ov_df[ov_df['asset'] == asset[0]], ov_labels[ov_df['asset'] == asset[0]]
            X, P, y, ind = trader.transform_data(df, labels)
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
                    quantity = int(amount * LEVERAGES[asset[1]] / open_price)

                    # Step 1 (morning) : decide what to do today and open position
                    pred, label = preds[i - 1], labels[ind[i - 1]]
                    order = decide_order(asset[1], quantity, open_price, pred, j)

                    # Step 2 (evening): close position
                    if order['is_buy'] is not None:
                        if order['is_buy']:
                            pl = round((close_price - order['open']) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)
                        else:
                            pl = round((order['open'] - close_price) * order['quantity'], 2)
                            # gpl = gross_pl(pl, quantity, now_close)

                    # Step 3 bis: save trade results
                    balance = round(balance + pl, 2)
                    bench_balance = round(bench_balance + (close_price - order['open']) * quantity, 2)
                    orders_hist.append(order['is_buy'])
                    balance_hist.append(balance)
                    bench_hist.append(bench_balance)
                    index_hist.append(ind[i])

            metrics = benchmark_metrics(INITIAL_GAMBLE, balance_hist, orders_hist)
            bench = round(bench_balance - INITIAL_GAMBLE, 2)
            profit = metrics['profit']
            pos_days = metrics['positive_days']
            m_returns = metrics['mean_returns']
            buy_acc, sell_acc = metrics['buy_accuracy'], metrics['sell_accuracy']
            m_wins, m_loss = metrics['mean_wins'], metrics['mean_loss']
            max_win, max_loss = metrics['max_win'], metrics['max_loss']

            assets_balance.append(balance_hist), benchmarks_balance.append(bench_hist)
            print(f'Profit: {profit}. Benchmark: {bench}. Mean daily return: {m_returns}%.')
            print(f'Correct moves: {pos_days}%. Correct buy: {buy_acc}%. Correct sell: {sell_acc}%.')
            print(f'Av. wins/loss amounts: {m_wins}/{m_loss}. Ext. wins/loss amounts: {max_win}/{max_loss}.')
            if plot:
                nice_plot(index_hist, [balance_hist, bench_hist], ['Algorithm', 'Benchmark'],
                          title=f'Profit evolution for {asset[1]}')

    print('_' * 100, '\n')
    if plot:
        portfolio_balance_hist = [sum([b[i] for b in assets_balance]) for i in range(len(assets_balance[0]))]
        benchmarks_balance_hist = [sum([b[i] for b in benchmarks_balance]) for i in range(len(benchmarks_balance[0]))]
        nice_plot(index_hist, [portfolio_balance_hist, benchmarks_balance_hist],
                  ['Portfolio balance', 'Benchmark balance'], title=f'Portfolio balance evolution')

    metrics = benchmark_portfolio_metric(INITIAL_GAMBLE, assets_balance)
    print('Returns:', metrics["assets_returns"])
    print(f'Average profit by assets: {metrics["assets_mean_profits"]}. '
          f'Average daily return: {metrics["assets_mean_returns"]}%.')
    print(f'Profitable assets: {metrics["count_profitable_assets"]}/{len(assets_balance)}.')
    print(f'Average daily profit of portfolio: {metrics["portfolio_mean_profits"]}. '
          f'Positive days: {metrics["portfolio_positive_days"]}%.')
    print('_' * 100, '\n')

    # perf = [(companies[i], ret) for i, ret in enumerate(assets_returns) if ret in sorted(assets_returns)[::-1][:11]]
    # perf = [companies[i] for i, ret in enumerate(assets_returns) if ret in sorted(assets_returns)[::-1][:11]]
    # print(perf)
    return metrics


def decide_order(asset, quantity, open_price, pred, date):
    # if pred_bid > (1 - tolerance) * now_ask:
    if pred == 1:
        order = {'asset': asset, 'is_buy': True, 'open': open_price, 'quantity': quantity, 'date': date}
    # elif pred_ask < now_bid / (1 - tolerance):
    elif pred == 0:
        order = {'asset': asset, 'is_buy': None, 'open': open_price, 'quantity': quantity, 'date': date}
    else:
        order = {'asset': asset, 'is_buy': None, 'open': open_price, 'quantity': quantity, 'date': date}
    return order


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


def update_data():
    folder = '../data/intrinio/'
    parser = configparser.ConfigParser()
    parser.read('../resources/intrinio.cfg')
    api_key = parser['INTRINIO']['access_token']
    for company in COMPANIES:
        path = folder + company.lower()
        fetch_intrinio_news(filename=path + '_news.csv', api_key=api_key, company=company, update=True)
        fetch_intrinio_prices(filename=path + '_prices.csv', api_key=api_key, company=company, update=True)
    precompute_embeddings(folder)


def get_recommendations():
    now = dt.now()
    folder = '../data/yahoo/'
    trader = LGBMTrader(load_from=f'Huorn_v{VERSION}')
    df, labels = load_data(folder, T0, T1)
    yesterday = df.index.max()
    df = df.loc[yesterday].reset_index(drop=True)
    # pd.DataFrame(df.loc[7]).T.to_csv('../outputs/report.csv', encoding='utf-8', mode='a')
    X, P, _, ind = trader.transform_data(df, labels)
    preds = trader.predict(X, P)
    reco, order_book = {'date': yesterday}, []
    lev = pd.Series([LEVERAGES[co] for co in COMPANIES])
    quantity = (INITIAL_GAMBLE * (lev / df['close'])).astype(int)
    for i, pred in enumerate(preds):
        if COMPANIES[i] in PERFORMERS:
            reco[COMPANIES[i]] = pred
            order = {'asset': COMPANIES[i], 'is_buy': pred, 'quantity': int(quantity[i]),
                     'data_date': yesterday, 'current_date': now.strftime("%Y-%m-%d %H:%M:%S")}
            order_book.append(order)
    path = '../outputs/recommendations.csv'
    reco = pd.DataFrame([reco]).set_index('date', drop=True)
    write_data(path, reco)
    logger.info(f'recommendations inference took {round((dt.now() - now).total_seconds())} seconds')
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


def safe_try(function, arg=None, max_attempts=999):
    """
    A safe environment to execute functions likely to fail (e.g. webdriver, scrapping...).

    :param function: function to safely execute.
    :param arg: one argument to feed the function (only one supported for now)
    :param max_attempts: maximum attempts for executing the function.
    :return: result of the function passed as argument.
    :rtype: Any
    """
    res, attempts = None, 0
    while attempts < max_attempts:
        try:
            if arg is None:
                res = function()
            else:
                res = function(arg)
        except Exception as ex:
            attempts += 1
            logger.warning(f"execution of function {function.__name__.strip()} failed {attempts} times."
                           f"Exception met: {ex}")
            continue
        break
    if attempts == max_attempts:
        logger.error(f"too many attempts to safely execute function {function.__name__.strip()}")
        raise Exception("Too many attempts to safely execute")
    return res


def grid_search():
    """
    Performs grid search to optimize a single parameter.
    :rtype: None
    """

    folder = '../data/intrinio/'
    df, labels = load_data(folder, T0, T1)
    trader = LGBMTrader(h=H, normalize=True)
    trader.ingest_traindata(df, labels)
    test_df, test_labels = load_data(folder, T0, T1, start_from=trader.t2)
    del df, labels

    res = []
    for param in [0.1, 0.2, 0.5, 0.9]:
        trader.lgb_params['learning_rate'] = param
        trader.train()
        trader.test(plot=False)
        trader.save(model_name=f'Huorn_v{VERSION}')
        metrics = backtest(plot=False, precomputed_df=test_df, precomputed_labels=test_labels)
        stats = {'num_iterations': param,
                 'mean_ret': metrics['assets_mean_returns'],
                 'pos_days': metrics['portfolio_positive_days'],
                 'prof_assets': metrics['count_profitable_assets']}
        print(f"-- Num iterations {param} --\t mean returns: {stats['mean_ret']}%\t"
              f"positive days: {stats['pos_days']}%\t profitable assets: {stats['prof_assets']}%\t")
        res.append(stats)
    print('\n\n\n\n', pd.DataFrame(res))


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
    # fetch_intrinio_data()
    # fetch_yahoo_data()
    # update_data()
    train_model()
    # backtest(plot=False)
    # grid_search()
    # o = get_recommendations()
    # place_orders(o)
    # get_trades_results()
    # yesterday_perf()
    # heartbeat()

    # for company in COMPANIES:
    #     folder = '../data/intrinio/'
    #     path = folder + company.lower()
    #     df = pd.read_csv(f'{path}_prices.csv', index_col=0)
    #     print(company, df.index.min())
