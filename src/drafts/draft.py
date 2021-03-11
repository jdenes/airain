"""
All developed functions that happen to be useless.
"""

import os
import logging
import requests
import numpy as np
import pandas as pd
# import tensorflow as tf

from datetime import datetime, timedelta
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

#############
# FROM MAIN #
#############

# def fetch_data():
#     print('Fetching data...')
#     con = fxcmpy.fxcmpy(config_file='../resources/fxcm.cfg', server='demo')
#     start, end = '2002-01-01 00:00:00', '2020-01-01 00:00:00'
#     fetch_fxcm_data(filename=f'../data/dataset_{LOWER_CURR}_train_{UNIT}{DATAFREQ}.csv',
#                     curr=CURR, unit=UNIT, start=start, end=end, freq=DATAFREQ, con=con)
#     start, end = '2020-01-01 00:00:00', '2020-06-01 00:00:00'
#     fetch_fxcm_data(filename=f'../data/dataset_{LOWER_CURR}_test_{UNIT}{DATAFREQ}.csv',
#                     curr=CURR, unit=UNIT, start=start, end=end, freq=DATAFREQ, con=con)

##############
# FROM UTILS #
##############


def fetch_crypto_rate(filename, from_currency, to_currency, start, end, freq):
    """
    Given currencies and start/end dates, as well as frequency, gets exchange rates from Poloniex API.
    """

    base_url = "https://poloniex.com/public?command=returnChartData"
    base_url += "&currencyPair=" + from_currency + "_" + to_currency
    data = []

    start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    tmp1 = start
    if end - start < timedelta(weeks=2 * int(freq)):
        tmp2 = end
    else:
        tmp2 = start + timedelta(weeks=2 * int(freq))
    while tmp2 <= end:
        x1, x2 = datetime.timestamp(tmp1), datetime.timestamp(tmp2)
        main_url = base_url + "&start=" + str(x1) + "&end=" + str(x2) + "&period=" + str(freq * 60)
        print('Fetching:', main_url)
        try:
            res = requests.get(main_url).json()
            if res[0]['date'] != 0:
                data += requests.get(main_url).json()
        except:
            raise ValueError('Unable to fetch data, please check connection and API availability.')

        tmp1, tmp2 = tmp1 + timedelta(weeks=2 * int(freq)), tmp2 + timedelta(weeks=2 * int(freq))
        if tmp1 < end < tmp2:
            tmp2 = end

    df = pd.DataFrame.from_dict(data).set_index('date')
    df.index = pd.to_datetime(df.index, unit='s')  # .tz_localize('UTC').tz_convert('Europe/Paris')
    df.to_csv(filename, encoding='utf-8')


def fetch_currency_rate(filename, from_currency, to_currency, freq, api_key):
    """
    Given a currency pair, gets all possible historic data from Alphavantage.
    """

    base_url = r"https://www.alphavantage.co/query?function=FX_INTRADAY"
    base_url += "&interval={}min&outputsize=full&apikey={}".format(str(freq), api_key)

    url = base_url + "&from_symbol=" + from_currency + "&to_symbol=" + to_currency
    print('Fetching:', url)
    data = requests.get(url).json()["Time Series FX (" + str(freq) + "min)"]
    df1 = pd.DataFrame.from_dict(data, orient='index')
    df1.columns = [(from_currency + to_currency + x[3:]) for x in df1.columns]

    url = base_url + "&from_symbol=" + to_currency + "&to_symbol=" + from_currency
    print('Fetching:', url)
    data = requests.get(url).json()["Time Series FX (" + str(freq) + "min)"]
    df2 = pd.DataFrame.from_dict(data, orient='index')
    df2.columns = [(to_currency + from_currency + x[3:]) for x in df2.columns]

    if not df1.index.equals(df2.index):
        print('Warning: index not aligned btw exchange rate and reverse exchange rate.')

    df = pd.concat([df1, df2], axis=1, sort=True).dropna()

    if os.path.exists(filename):
        old = pd.read_csv(filename, encoding='utf-8', index_col=0)
        df = pd.concat([old, df]).drop_duplicates(keep='last')

    df.to_csv(filename, encoding='utf-8')
    print('New available EUR-GBP data shape:', df.shape)


def fetch_fxcm_data(filename, curr, freq, con, unit='m', start=None, end=None, n_last=None):
    """
    Given currencies and start/end dates, as well as frequency, gets exchange rates from FXCM API.
    """

    if n_last is not None:
        df = con.get_candles(curr, period=unit + str(freq), number=n_last)
        df.index = pd.to_datetime(df.index, unit='s')
        df.to_csv(filename, encoding='utf-8')

    else:
        count = 0
        step = (1 + 0.5 * int(freq != 1)) * freq
        start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        tmp1 = start
        if end - start < timedelta(weeks=step):
            tmp2 = end
        else:
            tmp2 = start + timedelta(weeks=step)
        while tmp2 <= end:
            df = con.get_candles(curr, period=unit + str(freq), start=tmp1, stop=tmp2)
            df.index = pd.to_datetime(df.index, unit='s')
            if count == 0:
                df.to_csv(filename, encoding='utf-8')
            else:
                df.to_csv(filename, encoding='utf-8', mode='a', header=False)
            tmp1, tmp2 = tmp1 + timedelta(weeks=step), tmp2 + timedelta(weeks=step)
            count += 1
            if tmp1 < end < tmp2:
                tmp2 = end


def change_accuracy(y_true, y_pred):
    pred_shift = np.array(y_pred[1:] > y_true[:-1]).astype(int)
    true_shift = np.array(y_true[1:] > y_true[:-1]).astype(int)
    return (pred_shift == true_shift).mean()


def compute_metrics(y_true, y_pred):
    """
    Given true labels and predictions, outputs a set of performance metrics for regression task.
    """

    i = y_true != 0
    are = np.abs(y_pred[i] - y_true[i]) / np.abs(y_true[i])
    are = are[are < 1e8]
    r2 = r2_score(y_true, y_pred)
    me = max_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ca = change_accuracy(y_true, y_pred)

    return {'max_abs_rel_error': are.max(),
            'mean_abs_rel_error': are.mean(),
            'mean_absolute_error': mae,
            'max_error': me,
            'mean_squared_error': mse,
            'r2': r2,
            'change_accuracy': ca
            }


def evaluate(portfolio, rate):
    """
    Given a portfolio in units of base and quote currencies, returns value in base currency.
    """
    return round(portfolio[0] + (portfolio[1] * rate), 5)


def gross_pl(pl, K, price):
    return round((K / 10) * pl * (1 / price), 2)


def merge_finance_csv(folder='../data/finance', filename='../data/finance/global.csv'):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    res = []
    for i, f in enumerate(files):
        df = pd.read_csv(os.path.join(folder, f), encoding='utf-8', index_col=0)
        df['asset'] = f[:-4]
        res.append(df)
    res = pd.concat(res, axis=0, ignore_index=True)
    # df = df.set_index('Date')
    # df.index = pd.to_datetime(df.index, unit='s')
    res.to_csv(filename, encoding='utf-8')

################
# FROM TRADERS #
################

# def tensor_transform(self, tensor):
#     return tf.cast(tf.reshape(tensor, [-1]), dtype=tf.float32)
#
# def change_accuracy(self, y_true, y_pred):
#     y_true = self.tensor_transform(y_true)
#     y_pred = self.tensor_transform(y_pred)
#     x = tf.cast(((y_true * y_pred) > 0), dtype=tf.float32)
#     res = tf.math.reduce_mean(x)
#     return res
#
# def change_mean(self, y_true, y_pred):
#     pred_shift = tf.math.greater(y_pred[1:], y_true[:-1])
#     true_shift = tf.math.greater(y_true[1:], y_true[:-1])
#     same = tf.cast(true_shift, dtype=tf.float32)
#     res = tf.math.reduce_mean(same)
#     return res
#
# def sigma(self, y_true, y_pred):
#     y_true = self.tensor_transform(y_true)
#     y_pred = self.tensor_transform(y_pred)
#     x = y_true * y_pred
#     res = tf.math.reduce_mean(x) / tf.math.reduce_std(x)
#     return res

# labels, count = np.unique(self.y_train, return_counts=True)
# class_weight = {}
# for i, l in enumerate(labels):
#     class_weight[int(l)] = (1 / count[i]) * len(self.y_train) / len(labels)

# input_layer = tf.keras.layers.Input(shape=self.X_train.shape[-2:], name='input_X')
# price_layer = tf.keras.layers.Input(shape=self.P_train.shape[-1], name='input_P')

# lstm_layer = tf.keras.layers.LSTM(164, name='lstm')(input_layer)
# attention_layer = tf.keras.layers.Attention(name='attention_layer')([bilstm_layer, bilstm_layer])
# drop1_layer = tf.keras.layers.Dropout(0.1, name='dropout_1')(lstm_layer)

# concat_layer = tf.keras.layers.concatenate([lstm_layer, price_layer], name='concat')
# dense_layer = tf.keras.layers.Dense(20, name='combine')(concat_layer)
# drop2_layer = tf.keras.layers.Dropout(0.1, name='dropout_2')(dense_layer)
# output_layer = tf.keras.layers.Dense(2, name='output')(drop2_layer)
# self.model = tf.keras.Model(inputs=[input_layer, price_layer], outputs=output_layer)
# print(self.model.summary())

# self.model.compile(optimizer='adam', loss='mae')
# self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint = tf.keras.callbacks.ModelCheckpoint('../models/checkpoint.hdf5', monitor='val_loss',
# save_best_only=True)
# self.model.fit(train_data,
#                epochs=self.epochs,
#                steps_per_epoch=self.steps,
#                validation_steps=self.valsteps,
#                validation_data=val_data,
#                callbacks=[checkpoint]
#                )


# def grid_search():
#     """
#     Performs grid search to optimize a single parameter.
#     :rtype: None
#     """
#
#     folder = '../data/intrinio/'
#     df, labels = load_data(folder, T0, T1)
#     trader = LGBMTrader(h=H, normalize=True)
#     trader.ingest_traindata(df, labels)
#     test_df, test_labels = load_data(folder, T0, T1, start_from=trader.t2)
#     del df, labels
#
#     res = []
#     for param in [0.1, 0.2, 0.5, 0.9]:
#         trader.lgb_params['learning_rate'] = param
#         trader.train()
#         trader.test(plot=False)
#         trader.save(model_name=f'Huorn_v{VERSION}')
#         metrics = backtest(plot=False, precomputed_df=test_df, precomputed_labels=test_labels)
#         stats = {'num_iterations': param,
#                  'mean_ret': metrics['assets_mean_returns'],
#                  'pos_days': metrics['portfolio_positive_days'],
#                  'prof_assets': metrics['count_profitable_assets']}
#         print(f"-- Num iterations {param} --\t mean returns: {stats['mean_ret']}%\t"
#               f"positive days: {stats['pos_days']}%\t profitable assets: {stats['prof_assets']}%\t")
#         res.append(stats)
#     print('\n\n\n\n', pd.DataFrame(res))