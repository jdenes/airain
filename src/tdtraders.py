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

from traders import Trader

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
        model_name = '../models/' + model_name
        if self.model is not None:
            jl.dump(self.model, model_name + '/model.joblib')

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """
        super().load(model_name=model_name, fast=fast)
        model_name = '../models/' + model_name
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
