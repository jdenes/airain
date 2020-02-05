import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils import compute_metrics

####################################################################################


class Trader(object):
    """
    A general trader-forecaster to inherit from.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=True):
        """
        Initialize method.
        """

        self.h = h
        self.forecast = forecast
        self.seed = seed
        self.normalize = normalize

        self.model = None
        self.x_max, self.x_min = None, None
        self.y_max, self.y_min = None, None

        self.testsize = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def transform_data(self, df, labels, get_current=False):
        """
        Converts .csv file to appropriate data format for training for this agent type.
        """
        return None, None, None

    def ingest_traindata(self, df, labels, testsize=0.1):
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
        y_pred = self.predict(self.X_test)
        y_test = self.y_test

        if self.normalize:
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

    def backtest(self, df, labels, initial_gamble=1000, fees=0.01):
        """
        Given a dataset of any accepted format, simulates and returns portfolio evolution.
        /!\ WORKING ON CRYPTO DATA ONLY FOR NOW.
        """

        X, y, price = self.transform_data(df, labels, get_current=True)
        y_pred = self.predict(X)

        # TODO: policy should be to change only if gain even with the fees
        policy = (y_pred * (1 - fees) > price).astype(int)
        next_compo = [(0, 1) if p == 1 else (1, 0) for p in policy]

        def evaluate(x, p): return x[0] + (x[1] * p)

        ppp = [{'portfolio': (initial_gamble, 0), 'value': initial_gamble}]
        for i in range(len(price)):
            last_portfolio = ppp[-1]['portfolio']
            value = evaluate(last_portfolio, price[i]) * (1 - fees)
            if next_compo[i][0] != last_portfolio[0] and next_compo[i][1] != last_portfolio[1]:
                value = value * (1 - fees)
            next_portfolio = (next_compo[i][0] * value, next_compo[i][1] * value / price[i])
            ppp.append({'portfolio': next_portfolio, 'value': value})

        return pd.DataFrame(ppp)

####################################################################################


class Dummy(Trader):

    def __init__(self):
        super().__init__()

    def backtest(self, df, labels, initial_gamble=1000, fees=0.01):

        # price = df['EURGBPclose']
        price = df['weightedAverage'][self.h-1:]

        next_compo = []
        for _ in range(len(price)):
            next_compo.append((0, 1))

        def evaluate(x, p): return x[0] + (x[1] * p)

        ppp = [{'portfolio': (initial_gamble, 0), 'value': initial_gamble}]
        for i in range(len(price)):
            last_portfolio = ppp[-1]['portfolio']
            value = evaluate(last_portfolio, price[i]) * (1 - fees)
            if next_compo[i][0] != last_portfolio[0] and next_compo[i][1] != last_portfolio[1]:
                value = value * (1 - fees)
            next_portfolio = (next_compo[i][0] * value, next_compo[i][1] * value / price[i])
            ppp.append({'portfolio': next_portfolio, 'value': value})

        return pd.DataFrame(ppp)

####################################################################################


class Randommy(Trader):

    def __init__(self):
        super().__init__()

    def backtest(self, df, labels, initial_gamble=1000, fees=0.01):

        # price = df['EURGBPclose']
        price = df['weightedAverage'][self.h-1:]

        next_compo = []
        for _ in range(len(price)):
            next_compo.append([(0, 1), (1, 0)][np.random.choice(2)])

        def evaluate(x, p): return x[0] + (x[1] * p)

        ppp = [{'portfolio': (initial_gamble, 0), 'value': initial_gamble}]
        for i in range(len(price)):
            last_portfolio = ppp[-1]['portfolio']
            value = evaluate(last_portfolio, price[i]) * (1 - fees)
            if next_compo[i][0] != last_portfolio[0] and next_compo[i][1] != last_portfolio[1]:
                value = value * (1 - fees)
            next_portfolio = (next_compo[i][0] * value, next_compo[i][1] * value / price[i])
            ppp.append({'portfolio': next_portfolio, 'value': value})

        return pd.DataFrame(ppp)

####################################################################################


class LstmTrader(Trader):
    """
    A trader-forecaster based on a LSTM neural network.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=True):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize)

        self.batch_size = None
        self.buffer_size = None
        self.epochs = None
        self.steps = None
        self.valsteps = None
        self.gpu = None

        self.valsize = None
        self.X_val = None
        self.y_val = None

    def transform_data(self, df, labels, get_current=False):
        """
        Given data and labels, transforms it into suitable format and return them.
        """
        current = df['weightedAverage'].reset_index(drop=True)
        # current = df['EURGBPclose'].reset_index(drop=True)
        df, labels = df.reset_index(drop=True), labels.reset_index(drop=True)

        if self.normalize:
            df = 2 * (df - self.x_min) / (self.x_max - self.x_min) - 1
            labels = 2 * (labels - self.y_min) / (self.y_max - self.y_min) - 1

        df, labels = df.to_numpy(), labels.to_numpy()
        X, y, c = [], [], []

        for i in range(self.h-1, len(df)):
            ind = [int(i - self.h + x + 1) for x in range(self.h)]
            X.append(df[ind])
            y.append(labels[i])
            c.append(current[i])

        if get_current:
            return np.array(X), np.array(y), np.array(c)
        else:
            return np.array(X), np.array(y), None

    def ingest_traindata(self, df, labels, testsize=0.1, valsize=0.1):
        """
        Loads data from csv file depending on data type.
        """

        self.testsize = testsize
        self.valsize = valsize
        self.x_max, self.x_min = df.max(axis=0), df.min(axis=0)
        self.y_min, self.y_max = labels.min(), labels.max()

        X, y, _ = self.transform_data(df, labels)
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

####################################################################################


class MlTrader(Trader):
    """
    A trader-forecaster based on a traditional machine learning algorithm.
    """

    def transform_data(self, df, labels, get_current=False):
        """
        Given data and labels, transforms it into suitable format and return them.
        """

        current = df['weightedAverage']
        # current = df['EURGBPclose']

        if self.normalize:
            df = 2 * (df - self.x_min) / (self.x_max - self.x_min) - 1
            labels = 2 * (labels - self.y_min) / (self.y_max - self.y_min) - 1

        history = pd.DataFrame()
        for i in range(1, self.h):
            shifted_df = df.shift(i)
            history = pd.concat([history, shifted_df], axis=1)
        df = pd.concat([df, history], axis=1)
        df['labels'], df['current'] = labels, current
        df = df.dropna()

        X = df.drop(['labels', 'current'], axis=1).to_numpy()
        y = df['labels'].to_numpy()
        c = df['current'].to_numpy()

        if get_current:
            return X, y, c
        else:
            return X, y, None

    def ingest_traindata(self, df, labels, testsize=0.1):
        """
        Loads data from csv file depending on data type.
        """

        self.testsize = testsize
        self.x_max, self.x_min = df.max(axis=0), df.min(axis=0)
        self.y_min, self.y_max = labels.min(), labels.max()

        X, y, _ = self.transform_data(df, labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testsize)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print('Available data shapes: train:', self.X_train.shape, ' and test:', self.X_test.shape)

####################################################################################


class ForestTrader(MlTrader):
    """
    A trader-forecaster based on random forest.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=True):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize)
        self.n_estimators = None

    def train(self, n_estimators=10):
        """
        Using prepared data, trains model depending on agent type.
        """

        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(n_estimators=self.n_estimators)
        self.model.fit(self.X_train, self.y_train)

####################################################################################


class SvmTrader(MlTrader):
    """
    A trader-forecaster based on support vector regression.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=True):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize)
        self.kernel = None
        self.gamma = None

    def train(self, gamma='scale', kernel='rbf'):
        """
        Using prepared data, trains model depending on agent type.
        """

        self.kernel = kernel
        self.gamma = gamma
        self.model = svm.SVR(gamma='scale', kernel='rbf')
        self.model.fit(self.X_train, self.y_train)

####################################################################################


class NeuralTrader(MlTrader):
    """
    A trader-forecaster based on simple ANN.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=True):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize)
        self.layers = None

    def train(self, layers=(50, 100, 500, 50)):
        """
        Using prepared data, trains model depending on agent type.
        """

        self.layers = layers
        self.model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=self.layers)
        self.model.fit(self.X_train, self.y_train)
