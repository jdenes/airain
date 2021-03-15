
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

            print(len(df))
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

