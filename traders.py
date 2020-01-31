import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from tqdm import tqdm

from utils import compute_metrics


class Trader(object):
    """
    A general trader-forecaster to inherit from.
    """

    def __init__(self, freq=5, h=10, seed=123, forecast=1, datatype='crypto', normalize=True):
        """
        Initialize method.
        """

        self.freq = freq
        self.h = h
        self.forecast = forecast
        self.seed = seed
        self.datatype = datatype
        self.normalize = normalize

        self.model = None
        self.x_max, self.x_min = None, None
        self.y_max, self.y_min = None, None

        self.testsize = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def transform_data(self, df, labels):
        """
        Converts .csv file to appropriate data format for training for this agent type.
        """
        pass

    def ingest_traindata(self, df, labels, testsize):
        """
        Converts data file and ingest for training, depending on agent type.
        """
        pass

    def train(self):
        """
        Using prepared data, trains model depending on agent type.
        """
        pass

    def test(self, plot=False):
        """
        Once model is trained, uses test data to output performance metrics.
        """
        pass

    def predict(self, X):
        """
        Once the model is trained, predicts output if given appropriate data.
        """
        pass

    def backtest(self, data, initial_gamble):
        """
        Given a dataset of any accepted format, simulates and returns portfolio evolution.
        """
        pass


class LstmTrader(Trader):
    """
    A trader-forecaster based on a LSTM neural network.
    """

    def __init__(self, freq=5, h=10, seed=123, forecast=1, datatype='crypto', normalize=True):
        """
        Initialize method.
        """

        super().__init__(freq=freq, h=h, seed=seed, forecast=forecast, datatype=datatype, normalize=normalize)

        self.batch_size = None
        self.buffer_size = None
        self.epochs = None
        self.steps = None
        self.valsteps = None
        self.gpu = None

        self.valsize = None
        self.X_val = None
        self.y_val = None

    def transform_data(self, df, labels):
        """
        Given data and labels, transforms it into suitable format and return them.
        """

        if self.normalize:
            df = 2 * (df - self.x_min) / (self.x_max - self.x_min) - 1
            labels = 2 * (labels - self.y_min) / (self.y_max - self.y_min) - 1

        X, y = [], []

        for i, row in tqdm(df.iterrows()):
            end = datetime.strptime(i, '%Y-%m-%d %H:%M:%S')
            if self.datatype in ['short_currency', 'crypto']:
                ind = [str(end - timedelta(minutes=x*self.freq)) for x in range(self.h)]
            else:
                ind = [str(end - timedelta(days=x)) for x in range(self.h)]
            if all(x in df.index for x in ind):
                slicing = df.loc[ind]
                X.append(np.array(slicing))
                y.append(labels[i])

        return np.array(X), np.array(y)

    def ingest_traindata(self, df, labels, testsize=0.1, valsize=0.1):
        """
        Loads data from csv file depending on datatype.
        """

        self.testsize = testsize
        self.valsize = valsize
        self.x_max, self.x_min = df.max(axis=0), df.min(axis=0)
        self.y_min, self.y_max = labels.min(), labels.max()

        X, y = self.transform_data(df, labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testsize)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.valsize)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def train(self, batch_size=1000, buffer_size=10000, epochs=20, steps=200, valsteps=50, gpu=True):
        """
        Using prepared data, trains model depending on agent type.
        """

        self.gpu = gpu
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.steps = steps
        self.valsteps = valsteps

        if not self.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        train_data = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_data = train_data.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_data = val_data.batch(self.batch_size).repeat()

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(50, input_shape=self.X_train.shape[-2:], return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

        self.model.fit(train_data,
                       epochs=self.epochs,
                       steps_per_epoch=self.steps,
                       validation_steps=self.valsteps,
                       validation_data=val_data)

    def test(self, plot=False):
        """
        Once model is trained, uses test data to output performance metrics.
        """
        y_pred = self.model.predict(self.X_test).flatten()
        y_test = self.y_test

        if self.normalize:
            y_pred = (y_pred + 1) * (self.y_max - self.y_min) / 2 + self.y_min
            y_test = (y_test + 1) * (self.y_max - self.y_min) / 2 + self.y_min

        if plot:
            plt.plot((y_pred - y_test) / y_test, '.')
            plt.show()
            plt.plot(y_test, y_pred, '.')
            plt.show()

        return compute_metrics(y_test, y_pred)

    def predict(self, X):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        y_pred = self.model.predict(X).flatten()
        if self.normalize:
            y_pred = (y_pred + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        return y_pred

    def backtest(self, data, initial_gamble):
        """
        Given a dataset of any accepted format, simulates and returns portfolio evolution.
        """
        pass


class MlTrader(Trader):
    """
    A trader-forecaster based on a traditional machine learning algorithm.
    """

    def transform_data(self, df, labels):
        """
        Given data and labels, transforms it into suitable format and return them.
        """

        if self.normalize:
            df = 2 * (df - self.x_min) / (self.x_max - self.x_min) - 1
            labels = 2 * (labels - self.y_min) / (self.y_max - self.y_min) - 1

        history = pd.DataFrame()
        for i in range(1, self.h):
            shifted_df = df.shift(i)
            history = pd.concat([history, shifted_df], axis=1, sort=True)
        df = pd.concat([df, history], axis=1, sort=True)
        df['labels'] = labels
        df = df.dropna()

        X = df.loc[:, df.columns != 'labels'].to_numpy()
        y = df['labels'].to_numpy()

        return X, y

    def ingest_traindata(self, df, labels, testsize=0.1):
        """
        Loads data from csv file depending on data type.
        """

        self.testsize = testsize
        self.x_max, self.x_min = df.max(axis=0), df.min(axis=0)
        self.y_min, self.y_max = labels.min(), labels.max()

        X, y = self.transform_data(df, labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testsize)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        """
        Using prepared data, trains model depending on agent type.
        """

        self.model = RandomForestRegressor(n_estimators=10)
        # self.model = svm.SVR(gamma='scale', kernel='rbf')
        # self.model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 1000, 500, 50))
        self.model.fit(self.X_train, self.y_train)

    def test(self, plot=False):
        """
        Once model is trained, uses test data to output performance metrics.
        """
        y_pred = self.model.predict(self.X_test).flatten()
        y_test = self.y_test

        if self.normalize:
            y_pred = (y_pred + 1) * (self.y_max - self.y_min) / 2 + self.y_min
            y_test = (y_test + 1) * (self.y_max - self.y_min) / 2 + self.y_min

        if plot:
            plt.plot((y_pred - y_test) / y_test, '.')
            plt.show()
            plt.plot(y_test, y_pred, '.')
            plt.show()

        return compute_metrics(y_test, y_pred)

    def predict(self, X):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        y_pred = self.model.predict(X).flatten()
        if self.normalize:
            y_pred = (y_pred + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        return y_pred

    def backtest(self, data, initial_gamble):
        """
        Given a dataset of any accepted format, simulates and returns portfolio evolution.
        """
        pass
