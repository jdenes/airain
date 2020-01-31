from traders import LstmTrader, MlTrader
from utils import load_data, fetch_crypto_rate, fetch_exchange_rate, structure_crypto, structure_currencies


if __name__ == "__main__":

    freq = 5
    end = '2020-01-27 00:00:00'
    start = '2018-07-10 00:00:00'
    h = 20
    api_key = "H2T4H92C43D9DT3D"
    from_curr = 'USDC'
    to_curr = 'BTC'

    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=X.shape[-2:]))
    # model.add(tf.keras.layers.RepeatVector(1))
    # model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))
    # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu')))
    # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
    # model.compile(loss='mse', optimizer='adam')

    if False:
        for x, y in [('EUR', 'GBP'), ('GBP', 'EUR')]:
            fetch_exchange_rate(x, y, freq, api_key)
        structure_currencies(from_curr, to_curr, freq)

    if False:
        fetch_crypto_rate(from_curr, to_curr, start, end, freq)
        structure_crypto(from_curr, to_curr, freq)

    # score = hard_prediction(h=h, data='crypto', freq=freq, gpu=True)
    # score = simple_prediction(h=h, data='crypto')
    # print("Final model scores:", score)

    df, labels = load_data(datatype='crypto')
    # trader = LstmTrader()
    trader = MlTrader()
    trader.ingest_traindata(df, labels)
    # trader.train(epochs=2, steps=50)
    trader.train()
    score = trader.test(plot=True)
    print(score)
