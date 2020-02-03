from traders import LstmTrader, NeuralTrader, SvmTrader, ForestTrader
from utils import load_data, fetch_crypto_rate, fetch_currency_rate
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    freq = 5
    h = 20
    api_key = "H2T4H92C43D9DT3D"
    from_curr, to_curr = 'USDC', 'BTC'

    # start, end = '2018-07-01 00:00:00', '2019-11-30 00:00:00'
    # fetch_crypto_rate('./data/dataset_crypto_train.csv', from_curr, to_curr, start, end, freq)
    # start, end = '2019-12-01 00:00:00', '2020-02-01 00:00:00'
    # fetch_crypto_rate('./data/dataset_crypto_test.csv', from_curr, to_curr, start, end, freq)
    # fetch_currency_rate('./data/dataset_eurgbp.csv', 'EUR', 'GBP', freq, api_key)

    df, labels = load_data(filename='./data/dataset_crypto_train.csv', datatype='crypto', shift=1)
    print(df.shape, labels.shape)
    df1, labels1 = load_data(filename='./data/dataset_crypto_test.csv', datatype='crypto', shift=1)
    scores = []

    for trader_model in [ForestTrader]:
        trader = trader_model()
        trader.ingest_traindata(df, labels)
        # trader.train(epochs=40, steps=500)
        trader.train()
        scores.append(trader.test(plot=False))
        backtest = trader.backtest(df1, labels1, 1000, 0.0001)
        plt.plot(backtest['value'])
        plt.show()

    print(tabulate(pd.DataFrame(scores, index=[0]), headers="keys", tablefmt="fancy_grid"))
    backtest['value'].to_csv('./figures/wowmoney.csv', header=True)
