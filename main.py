from traders import NeuralTrader, LstmTrader, SvmTrader, ForestTrader, Dummy, Randommy, IdealTrader
from utils import load_data, fetch_crypto_rate, fetch_currency_rate, fetch_fxcm_data
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def tuning_n_estimators():
    freq = 5
    h = 10
    initial_gamble = 100
    fees = 0.0
    scores = []
    df1, labels1, price1 = load_data(filename='./data/dataset_crypto_test.csv', target_col='weightedAverage', shift=1)

    baseline = Dummy().backtest(df1, labels1, price1, initial_gamble, fees)
    random = Randommy().backtest(df1, labels1, price1, initial_gamble, fees)

    import numpy as np
    for i in range(100, 201, 100):
        print("Size:", i)
        res = []
        for _ in range(1):
            trader_model = ForestTrader
            trader = trader_model(h=h)
            trader.ingest_traindata(df1, labels1)
            trader.train(n_estimators=i)
            scores.append(trader.test(plot=False))
            backtest = trader.backtest(df1, labels1, price1, initial_gamble, fees)
            res.append(backtest['value'])
        res = np.array(res).mean(0)
        plt.plot(res, label='Huorn ' + str(i))

    plt.plot(baseline['value'], label='Pure BTC')
    plt.plot(random['value'], label='Random')
    plt.legend()
    plt.grid()
    plt.show()

    print(tabulate(pd.DataFrame(scores), headers="keys", tablefmt="fancy_grid"))


if __name__ == "__main__":
    freq = 5
    h = 10
    initial_gamble = 1000
    fees = 0.0
    api_key = "H2T4H92C43D9DT3D"
    from_curr, to_curr = 'USDC', 'BTC'

    # start, end = '2018-07-01 00:00:00', '2019-11-30 00:00:00'
    # fetch_crypto_rate('./data/dataset_crypto_train.csv', from_curr, to_curr, start, end, freq)
    #
    # start, end = '2019-12-01 00:00:00', '2020-02-01 00:00:00'
    # fetch_crypto_rate('./data/dataset_crypto_test.csv', from_curr, to_curr, start, end, freq)
    #
    # fetch_currency_rate('./data/dataset_eurgbp.csv', 'EUR', 'GBP', 5, api_key)

    # df, labels, price = load_data(filename='./data/dataset_crypto_train.csv', target_col='weightedAverage', shift=1)
    # df1, labels1, price1 = load_data(filename='./data/dataset_crypto_test.csv', target_col='weightedAverage', shift=1)

    ##############################################################################
    # start, end = '2020-02-06 00:00:00', datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # fetch_crypto_rate('./data/dataset_crypto_now.csv', from_curr, to_curr, start, end, freq)
    # df2, labels2, price2 = load_data(filename='./data/dataset_crypto_now.csv',
    #                                  target_col='weightedAverage',
    #                                  shift=1,
    #                                  keep_last=True)
    # print("Current price:", price2.index[-1], price2.to_list()[-1])
    # res = trader.predict_next(df2, labels2, price2, value=initial_gamble, fees=fees)
    # print(res)
    ##############################################################################

    # tuning_n_estimators()

    # import fxcmpy
    # TOKEN = '9c9f8a5725072aa250c8bd222dee004186ffb9e0'
    # con = fxcmpy.fxcmpy(access_token=TOKEN, server='demo')
    #
    # start, end = '2018-11-30 00:00:00', '2019-11-30 00:00:00'
    # fetch_fxcm_data(filename='./data/dataset_eurusd_train.csv', start=start, end=end, freq=freq, con=con)
    #
    # start, end = '2019-12-01 00:00:00', '2020-02-01 00:00:00'
    # fetch_fxcm_data(filename='./data/dataset_eurusd_test.csv', start=start, end=end, freq=freq, con=con)
    #
    df, labels, price = load_data(filename='./data/dataset_eurusd_train.csv', target_col='bidclose', shift=1)
    df1, labels1, price1 = load_data(filename='./data/dataset_eurusd_test.csv', target_col='bidclose', shift=1)
    # df, labels, price = load_data(filename='./data/dataset_crypto_train.csv', target_col='high', shift=1)
    # df1, labels1, price1 = load_data(filename='./data/dataset_crypto_test.csv', target_col='high', shift=1)

    trader1 = LstmTrader(h=h)
    # trader1.ingest_traindata(df, labels)
    # trader1.train(epochs=50)
    trader1.load(model_name='LSTM bidclose')
    trader2 = ForestTrader(h=h)
    # trader2.ingest_traindata(df, labels)
    # trader2.train(n_estimators=100)
    trader2.load(model_name='Huorn bidclose')
    baseline = Dummy().backtest(df1, labels1, price1, initial_gamble, fees)
    random = Randommy().backtest(df1, labels1, price1, initial_gamble, fees)
    ideal = IdealTrader().backtest(df1, labels1, price1, initial_gamble, fees)
    backtest1 = trader1.backtest(df1, labels1, price1, initial_gamble, fees)
    backtest2 = trader2.backtest(df1, labels1, price1, initial_gamble, fees)
    plt.plot(baseline['value'], label='Pure USD')
    plt.plot(random['value'], label='Random')
    # plt.plot(ideal['value'], label='Ideal')
    plt.plot(backtest2['value'], label='Huorn')
    plt.plot(backtest1['value'], label='Lstm')
    plt.legend()
    plt.grid()
    plt.show()

    # from sklearn.tree import plot_tree
    # plot_tree(trader2.model.estimators_[0])
