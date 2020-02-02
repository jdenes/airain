from traders import LstmTrader, NeuralTrader, SvmTrader, ForestTrader
from utils import load_data, fetch_crypto_rate, fetch_exchange_rate, structure_crypto, structure_currencies
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    freq = 5
    h = 20
    api_key = "H2T4H92C43D9DT3D"
    from_curr = 'USDC'
    to_curr = 'BTC'

    if False:
        for x, y in [('EUR', 'GBP'), ('GBP', 'EUR')]:
            fetch_exchange_rate(x, y, freq, api_key)
        structure_currencies(from_curr, to_curr, freq)

    start = '2019-08-02 00:00:00'
    end = '2020-02-02 00:00:00'
    fetch_crypto_rate(from_curr, to_curr, start, end, freq)
    structure_crypto('./data/csv/dataset_crypto_test.csv', from_curr, to_curr, freq)

    # score = hard_prediction(h=h, data='crypto', freq=freq, gpu=True)
    # score = simple_prediction(h=h, data='crypto')
    # print("Final model scores:", score)

    df, labels = load_data(filename='./data/csv/dataset_crypto_train.csv', datatype='crypto', shift=1)
    df1, labels1 = load_data(filename='./data/csv/dataset_crypto_test.csv', datatype='crypto', shift=1)
    scores = []

    for trader_model in [ForestTrader]:
        trader = trader_model()
        trader.ingest_traindata(df, labels)
        # trader.train(epochs=40, steps=500)
        trader.train()
        scores.append(trader.test(plot=False))
        backtest = trader.backtest(df1, labels1, 100)
        plt.plot(backtest['value'])
        plt.show()

    print(tabulate(pd.DataFrame(scores, index=[0]), headers="keys", tablefmt="fancy_grid"))
    backtest['value'].to_csv('./figures/wowmoney.csv', header=True)
