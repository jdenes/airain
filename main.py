from traders import NeuralTrader, LstmTrader, XgboostTrader, ForestTrader, Dummy, Randommy, IdealTrader
from fxcmtraders import FxcmTrader
from utils import load_data, fetch_crypto_rate, fetch_currency_rate, fetch_fxcm_data
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == "__main__":
    freq = 1
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


    # import fxcmpy
    # TOKEN = '9c9f8a5725072aa250c8bd222dee004186ffb9e0'
    # con = fxcmpy.fxcmpy(access_token=TOKEN, server='demo')
    #
    # start, end = '2009-11-30 00:00:00', '2020-01-30 00:00:00'
    # fetch_fxcm_data(filename='./data/dataset_eurusd_train.csv', start=start, end=end, freq=freq, con=con)
    #
    # start, end = '2020-01-01 00:00:00', '2020-02-01 00:00:00'
    # fetch_fxcm_data(filename='./data/dataset_eurusd_test.csv', start=start, end=end, freq=freq, con=con)

    df, labels, price = load_data(filename='./data/dataset_eurusd_train.csv', target_col='askclose', shift=1)
    df1, labels1, price1 = load_data(filename='./data/dataset_eurusd_test.csv', target_col='askclose', shift=1)
    # df, labels, price = load_data(filename='./data/dataset_crypto_train.csv', target_col='close', shift=1)
    # df1, labels1, price1 = load_data(filename='./data/dataset_crypto_test.csv', target_col='close', shift=1)

    # trader = ForestTrader(h=h)
    # trader.load(model_name='Huorn askclose')
    # print(trader.test(plot=False))
    # backtest = trader.backtest(df1, labels1, price1, initial_gamble, fees)
    # plt.plot(backtest['value'], label='Huorn')

    # baseline = Dummy().backtest(df1, labels1, price1, initial_gamble, fees)
    # plt.plot(baseline['value'], label='Pure USD')
    # random = Randommy().backtest(df1, labels1, price1, initial_gamble, fees)
    # plt.plot(random['value'], label='Random')

    # plt.legend()
    # plt.grid()
    # plt.show()

    master_trader = FxcmTrader(ask_model='Huorn askclose', bid_model='Huorn bidclose')
    backtest = master_trader.backtest(df1, labels1, price1, initial_gamble, fees)

