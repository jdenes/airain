import os
import json
import numpy as np
import pandas as pd
import joblib as jl
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from utils import compute_metrics, evaluate, normalize_data, unnormalize_data
from datetime import datetime as dt

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

    def transform_data(self, df, labels, get_index=False, keep_last=False):
        """
        Converts dataframe file to appropriate data format for this agent type.
        """
        if get_index:
            return None, None, np.array(0)
        else:
            return None, None

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
        # y_pred = self.predict(self.X_test)
        # y_test = self.y_test

        return self.model.evaluate(self.X_test, self.y_test)

        # if self.normalize:
        #     y_test = unnormalize_data(y_test, self.y_max, self.y_min)
        #
        # if plot:
        #     cond = ((y_pred > 0) == (y_test > 0))
        #     plt.plot((y_pred - y_test) / y_test, '.')
        #     plt.show()
        #     plt.plot(y_test[~cond], y_pred[~cond], '.', color='red')
        #     plt.plot(y_test[cond], y_pred[cond], '.')
        #     plt.show()
        #
        # return compute_metrics(y_test, y_pred)

    def predict(self, X):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        y_pred = self.model.predict(X)  # .flatten()
        if self.normalize:
            pass
            # y_pred = unnormalize_data(y_pred, self.y_max, self.y_min)
        return y_pred

    def compute_policy(self, df, labels, price, shift, fees):
        """
        Given parameters, decides what to do at next steps based on predictive model.
        """
        X, _, ind = self.transform_data(df, labels, get_index=True)
        current_price = price[ind].to_numpy()
        y_pred = self.predict(X)
        print(compute_metrics(current_price[shift:], y_pred[:-shift]))
        going_up = (y_pred * (1 - fees) > current_price)
        policy = [(0, 1) if p else (1, 0) for p in going_up]
        return policy, ind

    def backtest(self, df, labels, price, tradefreq=1, lag=0, initial_gamble=1000, fees=0.00):
        """
        Given a dataset of any accepted format, simulates and returns portfolio evolution.
        """

        policy, ind = self.compute_policy(df, labels, price, tradefreq+lag, fees)
        price = price[ind].to_numpy()

        count, amount = 0, 0
        next_portfolio = (initial_gamble, 0)
        ppp = []
        value = initial_gamble

        for i in range(len(price) - 1):

            if dt.strptime(ind[i], '%Y-%m-%d %H:%M:%S').minute % tradefreq == 0:

                last_portfolio = next_portfolio
                last_value = value
                value = evaluate(last_portfolio, price[i])
                if value < last_value:
                    count += 1
                    amount += last_value - value
                policy_with_lag = policy[i-lag]
                if i > tradefreq + lag and policy_with_lag != policy[i-tradefreq-lag]:
                    value *= 1 - fees
                next_portfolio = (policy_with_lag[0] * value, policy_with_lag[1] * value / price[i])
                ppp.append({'index': ind[i], 'portfolio': next_portfolio, 'value': value})

        print("Total bad moves share:", count/len(price), "for amount lost:", amount)
        return pd.DataFrame(ppp)

    def predict_last(self, df, labels):
        """
        Predicts next value and consequently next optimal portfolio.
        """
        X, _, _ = self.transform_data(df, labels, get_index=True, keep_last=True)
        y_pred = self.predict(X)

        return y_pred[-1]

    def save(self, model_name):
        """
        Save model to folder.
        """

        model_name = './models/' + model_name
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        to_rm = ['model', 'x_max', 'x_min', 'X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val']
        attr_dict = {}
        for attr, value in self.__dict__.items():
            if attr not in to_rm:
                attr_dict[attr] = value

        with open(model_name + '/attributes.json', 'w') as file:
            json.dump(attr_dict, file)

        self.x_max.to_csv(model_name + '/x_max.csv', header=False)
        self.x_min.to_csv(model_name + '/x_min.csv', header=False)
        np.save(model_name + '/X_train.npy', self.X_train)
        np.save(model_name + '/y_train.npy', self.y_train)
        np.save(model_name + '/X_test.npy', self.X_test)
        np.save(model_name + '/y_test.npy', self.y_test)

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """

        model_name = './models/' + model_name
        with open(model_name + '/attributes.json', 'r') as file:
            self.__dict__ = json.load(file)

        self.x_max = pd.read_csv(model_name + '/x_max.csv', header=None, index_col=0, squeeze=True)
        self.x_min = pd.read_csv(model_name + '/x_min.csv', header=None, index_col=0, squeeze=True)

        if not fast:
            self.X_train = np.load(model_name + '/X_train.npy')
            self.y_train = np.load(model_name + '/y_train.npy')
            self.X_test = np.load(model_name + '/X_test.npy')
            self.y_test = np.load(model_name + '/y_test.npy')


####################################################################################


class Dummy(Trader):

    def __init__(self):
        super().__init__()

    def compute_policy(self, df, labels, price, shift, fees):
        """
        Given parameters, decides what to do at next steps based on predictive model.
        """
        tmp = price.reset_index()[self.h-1:]
        current_price, ind = tmp['price'].to_list(), tmp['date'].to_list()
        policy = [(0, 1) for _ in current_price]
        return policy, ind

####################################################################################


class Randommy(Trader):

    def __init__(self):
        super().__init__()

    def compute_policy(self, df, labels, price, shift, fees):
        """
        Given parameters, decides what to do at next steps based on predictive model.
        """
        tmp = price.reset_index()[self.h-1:]
        current_price, ind = tmp['price'].to_list(), tmp['date'].to_list()
        policy = [[(0, 1), (1, 0)][np.random.choice(2)] for _ in current_price]
        return policy, ind

####################################################################################


class IdealTrader(Trader):

    def __init__(self):
        super().__init__()

    def compute_policy(self, df, labels, price, shift, fees):
        """
        Given parameters, decides what to do at next steps based on predictive model.
        """
        tmp = price.reset_index()[self.h-1:]
        cp, ind = tmp['price'].to_list(), tmp['date'].to_list()
        policy = [(0, 1) if cp[i+1] > cp[i] else (1, 0) for i in range(len(cp) - 1)] + [(1, 0)]
        return policy, ind

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

    def transform_data(self, df, labels, get_index=False, keep_last=True):
        """
        Given data and labels, transforms it into suitable format and return them.
        """
        index = df.index.to_list()
        df, labels = df.reset_index(drop=True), labels.reset_index(drop=True)

        if self.normalize:
            df = normalize_data(df, self.x_max, self.x_min)
            # labels = normalize_data(labels, self.y_max, self.y_min)

        df, labels = df.to_numpy(), labels.to_numpy()
        X, y, ind = [], [], []

        for i in range(self.h-1, len(df)):
            indx = [int(i - self.h + x + 1) for x in range(self.h)]
            X.append(df[indx])
            y.append(labels[i])
            ind.append(index[i])

        X, y, ind = np.array(X), np.array(y), np.array(ind)
        y = y.reshape((len(y), 1))

        if get_index:
            return X, y, ind
        else:
            return X, y

    def ingest_traindata(self, df, labels, testsize=0.1, valsize=0.1):
        """
        Loads data from csv file depending on data type.
        """

        self.testsize = testsize
        self.valsize = valsize

        split = int(self.testsize * len(df))
        train_ind, test_ind = df.index[:-split], df.index[-split:]
        train_ind, val_ind = train_ind[:-split], train_ind[-split:]

        df_train, labels_train = df.loc[train_ind], labels.loc[train_ind]
        self.x_max, self.x_min = df_train.max(axis=0), df_train.min(axis=0)
        # self.y_min, self.y_max = labels_train.min(), labels_train.max()

        X_train, y_train = self.transform_data(df_train, labels_train)
        self.X_train = X_train
        self.y_train = y_train
        del X_train, y_train, df_train, labels_train

        df_test, labels_test = df.loc[test_ind], labels.loc[test_ind]
        X_test, y_test = self.transform_data(df_test, labels_test)
        self.X_test = X_test
        self.y_test = y_test
        del X_test, y_test, df_test, labels_test

        df_val, labels_val = df.loc[val_ind], labels.loc[val_ind]
        X_val, y_val = self.transform_data(df_val, labels_val)
        self.X_val = X_val
        self.y_val = y_val
        del X_val, y_val, df_val, labels_val

    def train(self, batch_size=1000, buffer_size=10000, epochs=50, steps=1000, valsteps=100, gpu=True):
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
        self.model.add(tf.keras.layers.LSTM(150, input_shape=self.X_train.shape[-2:]))  # return_sequences=True))
        # self.model.add(tf.keras.layers.LSTM(50, return_sequences=True))
        # self.model.add(tf.keras.layers.LSTM(75, activation='relu'))
        # self.model.add(tf.keras.layers.Dropout(0.5))
        # self.model.add(tf.keras.layers.Dense(20, activation='relu'))
        # self.model.add(tf.keras.layers.Dense(1, activation='linear'))
        self.model.add(tf.keras.layers.Dense(100))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        # self.model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mse')
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

        self.model.fit(train_data,
                       epochs=self.epochs,
                       steps_per_epoch=self.steps,
                       validation_steps=self.valsteps,
                       validation_data=val_data)

    def save(self, model_name):
        """
        Save model to folder.
        """
        super().save(model_name)
        model_name = './models/' + model_name
        if self.model is not None:
            self.model.save(model_name + '/model.h5')

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """
        super().load(model_name=model_name, fast=fast)
        model_name = './models/' + model_name
        self.model = tf.keras.models.load_model(model_name + '/model.h5')

####################################################################################


class MlTrader(Trader):
    """
    A trader-forecaster based on a traditional machine learning algorithm.
    """

    def transform_data(self, df, labels, get_index=False, keep_last=False):
        """
        Given data and labels, transforms it into suitable format and return them.
        """

        index = df.index.to_list()
        if self.normalize:
            df = normalize_data(df, self.x_max, self.x_min)
            labels = normalize_data(labels, self.y_max, self.y_min)

        history = pd.DataFrame()
        for i in range(1, self.h):
            shifted_df = df.shift(i)
            history = pd.concat([history, shifted_df], axis=1)
        df = pd.concat([df, history], axis=1)
        del history, shifted_df
        df['labels'], df['ind'] = labels, index

        first_idx = df.apply(lambda col: col.first_valid_index()).max()
        if keep_last:
            df = df.loc[first_idx:]
        else:
            last_idx = df.apply(lambda col: col.last_valid_index()).max()
            df = df.loc[first_idx:last_idx]

        X = df.drop(['labels', 'ind'], axis=1).to_numpy()
        y = df['labels'].to_numpy()
        ind = df['ind'].to_numpy()

        if get_index:
            return X, y, ind
        else:
            return X, y

    def ingest_traindata(self, df, labels, testsize=0.1):
        """
        Loads data from csv file depending on data type.
        """

        self.testsize = testsize

        split = int(self.testsize * len(df))
        train_ind, test_ind = df.index[:-split], df.index[-split:]

        df_train, labels_train = df.loc[train_ind], labels.loc[train_ind]
        self.x_max, self.x_min = df_train.max(axis=0), df_train.min(axis=0)
        self.y_min, self.y_max = labels_train.min(), labels_train.max()

        X_train, y_train = self.transform_data(df_train, labels_train)
        self.X_train = X_train
        self.y_train = y_train
        del X_train, y_train, df_train, labels_train

        df_test, labels_test = df.loc[test_ind], labels.loc[test_ind]
        X_test, y_test = self.transform_data(df_test, labels_test)
        self.X_test = X_test
        self.y_test = y_test
        del X_test, y_test, df_test, labels_test

    def save(self, model_name):
        """
        Save model to folder.
        """
        super().save(model_name=model_name)
        model_name = './models/' + model_name
        if self.model is not None:
            jl.dump(self.model, model_name + '/model.joblib')

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """
        super().load(model_name=model_name, fast=fast)
        model_name = './models/' + model_name
        self.model = jl.load(model_name + '/model.joblib')


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
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=100, n_jobs=8, verbose=2)
        self.model.fit(self.X_train, self.y_train)
        self.model.verbose = 0

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

####################################################################################


class XgboostTrader(MlTrader):
    """
    A trader-forecaster based on XGBoost.
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
        self.model = xgb.XGBRegressor(objective='reg:squarederror',
                                      n_estimators=self.n_estimators, n_jobs=3, verbose=2)
        self.model.fit(self.X_train, self.y_train)
