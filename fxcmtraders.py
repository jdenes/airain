import os
import json
import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt

from utils import compute_metrics, evaluate, normalize_data, unnormalize_data

####################################################################################


class FxcmTrader(object):
    """
    A general class to use pre-trained trader-forecasters and adapt to FXCM trading platform.
    """

    def __init__(self, ask_model, bid_model):
        """
        Initialize method.
        """

        self.normalize = None
        self.y_max = None
        self.y_min = None
        self.h = None

        model_name = './models/' + ask_model
        with open(model_name + '/attributes.json', 'r') as file:
            self.__dict__ = json.load(file)

        self.x_max = pd.read_csv(model_name + '/x_max.csv', header=None, index_col=0, squeeze=True)
        self.x_min = pd.read_csv(model_name + '/x_min.csv', header=None, index_col=0, squeeze=True)
        self.X_train = np.load(model_name + '/X_train.npy')
        self.y_train = np.load(model_name + '/y_train.npy')
        self.X_test = np.load(model_name + '/X_test.npy')
        self.y_test = np.load(model_name + '/y_test.npy')

        self.ask_model = jl.load('./models/' + ask_model + '/model.joblib')
        self.bid_model = jl.load('./models/' + bid_model + '/model.joblib')

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

    def predict_ask(self, X):
        """
        Predicts ask price if given appropriate (transformed) data.
        """
        y_pred = self.ask_model.predict(X).flatten()
        if self.normalize:
            y_pred = unnormalize_data(y_pred, self.y_max, self.y_min)
        return y_pred

    def predict_bid(self, X):
        """
        Predicts bid price if given appropriate (transformed) data.
        """
        y_pred = self.bid_model.predict(X).flatten()
        if self.normalize:
            y_pred = unnormalize_data(y_pred, self.y_max, self.y_min)
        return y_pred

    def compute_policy(self, df, labels, price, fees):
        """
        Given parameters, decides what to do at next steps based on predictive model.
        """
        X, _, ind = self.transform_data(df, labels, get_index=True)
        current_price = price[ind].to_numpy()
        y_pred = self.predict(X)
        print(compute_metrics(current_price[1:], y_pred[:-1]))
        going_up = (y_pred * (1 - fees) > current_price)
        policy = [(0, 1) if p else (1, 0) for p in going_up]
        return policy, ind

    def backtest(self, df, labels, price, initial_gamble=1000, fees=0.01):
        """
        Given a dataset of any accepted format, simulates and returns portfolio evolution.
        """

        policy, ind = self.compute_policy(df=df, labels=labels, price=price, fees=fees)
        price = price[ind].to_numpy()

        count, amount = 0, 0
        next_portfolio = (initial_gamble, 0)
        ppp = []
        value = initial_gamble

        for i in range(len(price)):
            last_portfolio = next_portfolio
            last_value = value
            value = evaluate(last_portfolio, price[i])
            if value < last_value:
                count += 1
                amount += last_value - value
            if i > 0 and policy[i] != policy[i-1]:
                value *= 1 - fees
            next_portfolio = (policy[i][0] * value, policy[i][1] * value / price[i])
            ppp.append({'index': ind[i], 'portfolio': next_portfolio, 'value': value})

        print("Total bad moves share:", count/len(price), "for amount lost:", amount)
        return pd.DataFrame(ppp)

    def predict_next(self, df, labels, price, value=1000, fees=0.01):
        """
        Predicts next value and consequently next optimal portfolio.
        """
        X, y, ind = self.transform_data(df, labels, get_index=True, keep_last=True)
        price = price[ind].to_list()
        y_pred = self.predict(X)

        going_up = y_pred[-1] * (1 - fees) > price[-1]
        next_policy = (0, 1) if going_up else (1, 0)
        next_portfolio = (next_policy[0] * value, next_policy[1] * value / price[-1])
        next_value = evaluate(next_portfolio, y_pred[-1])

        return {'index': ind[-1],
                'next_portfolio': next_portfolio,
                'next_policy': next_policy,
                'next_value': next_value
                }
