from traders import LstmTrader, NeuralTrader, SvmTrader, ForestTrader, Dummy, Randommy
from utils import load_data, fetch_crypto_rate, fetch_currency_rate
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    freq = 5
    h = 10
    initial_gamble = 100
    fees = 0.0
    api_key = "H2T4H92C43D9DT3D"
    from_curr, to_curr = 'USDC', 'BTC'

    start, end = '2018-07-01 00:00:00', '2019-10-31 00:00:00'
    # fetch_crypto_rate('./data/dataset_crypto_train.csv', from_curr, to_curr, start, end, freq)

    start, end = '2019-12-01 00:00:00', '2020-02-01 10:00:00'
    fetch_crypto_rate('./data/dataset_crypto_test.csv', from_curr, to_curr, start, end, freq)

    fetch_currency_rate('./data/dataset_eurgbp.csv', 'EUR', 'GBP', freq, api_key)

    df, labels = load_data(filename='./data/dataset_crypto_train.csv', shift=1)
    df1, labels1 = load_data(filename='./data/dataset_crypto_test.csv', shift=1)

    scores = []
    x = df1.index[h-2:]

    for i, trader_model in enumerate([ForestTrader]):
        trader = trader_model(h=h)
        trader.ingest_traindata(df, labels)
        trader.train()
        scores.append(trader.test(plot=False))
        backtest = trader.backtest(df1, labels1, initial_gamble, fees)
        plt.plot(backtest['value'], label=['Huorn', 'LSTM', 'SVM'][i])

    baseline = Dummy().backtest(df1, labels1, initial_gamble, fees)
    random = Randommy().backtest(df1, labels1, initial_gamble, fees)

    plt.plot(baseline['value'], label='Pure BTC')
    plt.plot(random['value'], label='Random')
    plt.legend()
    plt.grid()
    plt.show()

    print(tabulate(pd.DataFrame(scores), headers="keys", tablefmt="fancy_grid"))
    # print(backtest['value'], backtest['portfolio'])
