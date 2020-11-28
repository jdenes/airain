import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from utils import normalize_data, unnormalize_data
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

####################################################################################


class Trader(object):
    """
    A general trader-forecaster to inherit from.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=False, load_from=None):
        """
        Initialize method.
        """

        self.h = h
        self.forecast = forecast
        self.normalize = normalize
        self.seed = seed
        np.random.seed(self.seed)

        self.model = None
        self.x_max, self.x_min = None, None
        self.p_max, self.p_min = None, None
        self.y_max, self.y_min = None, None

        self.testsize = None
        self.X_train = None
        self.P_train = None
        self.y_train = None
        self.X_test = None
        self.P_test = None
        self.y_test = None

        self.t1, self.t2 = '2020-04-01', '2020-06-01'

        if load_from is not None:
            self.load(model_name=load_from)

    def transform_data(self, df, labels, get_index=False, keep_last=False):
        """
        Converts dataframe file to appropriate data format for this agent type.
        """
        if get_index:
            return None, None, np.array(0)
        else:
            return None, None

    def ingest_traindata(self, df, prices, labels, testsize=0.1):
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

        y_train, y_val, y_test = self.y_train, self.y_val, self.y_test
        # if self.normalize:
        #     y_train = unnormalize_data(y_train, self.y_max, self.y_min)
        #     y_test = unnormalize_data(y_test, self.y_max, self.y_min)

        print("Performance metrics on train...")
        # self.model.evaluate({'input_X': self.X_train, 'input_P': self.P_train}, self.y_train)
        y_pred = self.predict(self.X_train, self.P_train)
        print(classification_report(y_train, y_pred, digits=4))
        # print(tf.math.confusion_matrix(self.y_train, y_pred))
        # print(compute_metrics(y_train, y_pred))

        print("Performance metrics on val...")
        # self.model.evaluate({'input_X': self.X_test, 'input_P': self.P_test}, self.y_test)
        y_pred = self.predict(self.X_val, self.P_val)
        print(classification_report(y_val, y_pred, digits=4))
        # print(tf.math.confusion_matrix(self.y_test, y_pred))
        # print(compute_metrics(y_test, y_pred))

        print("Performance metrics on test...")
        # self.model.evaluate({'input_X': self.X_test, 'input_P': self.P_test}, self.y_test)
        y_pred = self.predict(self.X_test, self.P_test)
        print(classification_report(y_test, y_pred, digits=4))
        # print(tf.math.confusion_matrix(self.y_test, y_pred))
        # print(compute_metrics(y_test, y_pred))

        if plot:

            fig, ax = plt.subplots(1, 2, figsize=(14, 14))
            lgb.plot_importance(self.model[0], ax=ax[0], max_num_features=25, precision=0,
                                importance_type='split', title='In number of splits')
            lgb.plot_importance(self.model[0], ax=ax[1], max_num_features=25, precision=0,
                                importance_type='gain', title='In total splits gain')
            fig.tight_layout()
            plt.show()

            i = y_test != 0
            plt.plot((y_pred[i] - y_test[i]), '.')
            plt.show()
            plt.plot(y_test, y_pred, '.')
            plt.show()

    def predict(self, X, P):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        # y_pred = self.model.predict((X, P))
        y_pred = np.array([mod.predict(P) for mod in self.model]).mean(axis=0)
        # y_pred = 1. / (1. + np.exp(-y_pred))
        y_pred = y_pred.round(0).astype(int)
        # y_pred = y_pred.reshape((len(y_pred), 1))
        # if self.normalize:
        #     y_pred = unnormalize_data(y_pred, self.y_max, self.y_min)
        return y_pred

    def save(self, model_name):
        """
        Save model to folder.
        """

        model_name = '../models/' + model_name
        if not os.path.exists(model_name):
            os.makedirs(model_name)
            os.makedirs(model_name + '/tree/')

        to_rm = ['model', 'x_max', 'x_min', 'p_min', 'p_max', 'X_train', 'P_train', 'y_train',
                 'X_test', 'P_test', 'y_test', 'X_val', 'P_val', 'y_val']
        attr_dict = {}
        for attr, value in self.__dict__.items():
            if attr not in to_rm:
                if isinstance(value, np.integer):
                    value = int(value)
                attr_dict[attr] = value

        with open(model_name + '/attributes.json', 'w') as file:
            json.dump(attr_dict, file)

        self.x_max.to_csv(model_name + '/x_max.csv', header=False)
        self.x_min.to_csv(model_name + '/x_min.csv', header=False)
        self.p_max.to_csv(model_name + '/p_max.csv', header=False)
        self.p_min.to_csv(model_name + '/p_min.csv', header=False)
        np.save(model_name + '/X_train.npy', self.X_train)
        self.P_train.to_csv(model_name + '/P_train.csv')
        np.save(model_name + '/y_train.npy', self.y_train)
        np.save(model_name + '/X_test.npy', self.X_test)
        self.P_test.to_csv(model_name + '/P_test.csv')
        np.save(model_name + '/y_test.npy', self.y_test)

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """

        model_name = '../models/' + model_name
        with open(model_name + '/attributes.json', 'r') as file:
            self.__dict__ = json.load(file)

        self.x_max = pd.read_csv(model_name + '/x_max.csv', header=None, index_col=0, squeeze=True)
        self.x_min = pd.read_csv(model_name + '/x_min.csv', header=None, index_col=0, squeeze=True)
        self.p_max = pd.read_csv(model_name + '/p_max.csv', header=None, index_col=0, squeeze=True)
        self.p_min = pd.read_csv(model_name + '/p_min.csv', header=None, index_col=0, squeeze=True)

        if not fast:
            self.X_train = np.load(model_name + '/X_train.npy')
            self.P_train = pd.read_csv(model_name + '/P_train.csv', index_col=0)
            self.y_train = np.load(model_name + '/y_train.npy')
            self.X_test = np.load(model_name + '/X_test.npy')
            self.P_test = pd.read_csv(model_name + '/P_test.csv', index_col=0)
            self.y_test = np.load(model_name + '/y_test.npy')


####################################################################################


class LstmTrader(Trader):
    """
    A trader-forecaster based on a LSTM neural network.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=False, load_from=None):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize, load_from=load_from)

        self.batch_size = None
        self.buffer_size = None
        self.epochs = None
        self.steps = None
        self.valsteps = None
        self.gpu = None

        self.valsize = None
        self.X_val = None
        self.P_val = None
        self.y_val = None
        self.n_estimators = 1

        self.lgb_params = {
            'objective': 'binary',      # multiclass
            # 'num_class': 3,
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',            # dart, goss
            'subsample': 1.0,                   # 0.5
            'subsample_freq': 1,
            'learning_rate': 0.01,              # 0.03
            'top_rate': 0.7,
            'other_rate': 0.1,
            'num_leaves': 2 ** 12 - 1,
            'min_data_in_leaf': 2 ** 12 - 1,
            'feature_fraction': 0.7,            # between 0.4 and 0.6
            'max_bin': 11,                      # 255
            'num_iterations': 1140,             # 7000,
            'boost_from_average': True,
            'verbose': -1,
            # 'early_stopping_rounds': 300,
        }

    def transform_data(self, df, labels, get_index=False, keep_last=True):
        """
        Given data and labels, transforms it into suitable format and return them.
        """

        index = df.index.to_list()
        df, labels = df.reset_index(drop=True), labels.reset_index(drop=True)

        if self.normalize:
            df = normalize_data(df, self.x_max, self.x_min)
            # labels = normalize_data(labels, self.y_max, self.y_min)

        columns = df.columns
        df, labels = df.to_numpy(), labels.to_numpy()
        X, P, y, ind = [], [], [], []

        for i in range(len(df)):  # range(self.h-1, len(df))
            # indx = [int(i - self.h + x + 1) for x in range(self.h)]
            # X.append(df[indx])
            P.append(df[i])
            y.append(labels[i])
            ind.append(index[i])

        X, P, y, ind = np.array(X), np.array(P), np.array(y), np.array(ind)
        # y = y.reshape((len(y), 1))
        # y = to_categorical(y)
        P = pd.DataFrame(P, columns=columns)

        if get_index:
            return X, P, y, ind
        else:
            return X, P, y

    def ingest_traindata(self, df, labels, testsize=0.1, valsize=0.1):
        """
        Loads data from csv file depending on data type.
        """

        self.testsize = testsize
        self.valsize = valsize

        df_train, labels_train = df.loc[:self.t1], labels.loc[:self.t1]
        self.x_max, self.x_min = df_train.max(axis=0), df_train.min(axis=0)
        self.p_max, self.p_min = self.x_max, self.x_min
        self.y_min, self.y_max = labels_train.min(), labels_train.max()

        X, P, y = self.transform_data(df_train, labels_train)
        if self.X_train is None:
            self.X_train, self.P_train, self.y_train = X, P, y
        else:
            self.X_train = np.concatenate((self.X_train, X))
            self.P_train = pd.concat((self.P_train, P), axis=0)
            self.y_train = np.concatenate((self.y_train, y))
        del df_train, labels_train

        df_test, labels_test = df.loc[self.t2:], labels.loc[self.t2:]
        X, P, y = self.transform_data(df_test, labels_test)
        if self.X_test is None:
            self.X_test, self.P_test, self.y_test = X, P, y
        else:
            self.X_test = np.concatenate((self.X_test, X))
            self.P_test = pd.concat((self.P_test, P), axis=0)
            self.y_test = np.concatenate((self.y_test, y))
        del df_test, labels_test

        df_val, labels_val = df.loc[self.t1:self.t2], labels.loc[self.t1:self.t2]
        X, P, y = self.transform_data(df_val, labels_val)
        if self.X_val is None:
            self.X_val, self.P_val, self.y_val = X, P, y
        else:
            self.X_val = np.concatenate((self.X_val, X))
            self.P_val = pd.concat((self.P_val, P), axis=0)
            self.y_val = np.concatenate((self.y_val, y))
        del df_val, labels_val

    def train(self, batch_size=80, buffer_size=10000, epochs=50, steps=1000, valsteps=100, gpu=True):
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
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        print('_' * 100, '\n')
        print(f'Training on {self.P_train.shape[0]} examples, with {self.P_train.shape[1]} features...')
        y_mean = (self.y_train > 0).mean()
        print(f'Baseline for change accuracy is {round(max(y_mean, 1 - y_mean), 4)}.')
        print('_' * 100, '\n')

        # train_data = tf.data.Dataset.from_tensor_slices(
        #                           ({'input_X': self.X_train, 'input_P': self.P_train}, self.y_train))
        # train_data = train_data.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        # val_data = tf.data.Dataset.from_tensor_slices(
        #                           ({'input_X': self.X_val, 'input_P': self.P_val}, self.y_val))
        # val_data = val_data.shuffle(self.buffer_size).batch(self.batch_size).repeat()

        self.model = []
        cat_feat = ['asset']
        for i in range(self.n_estimators):
            # idx = (np.random.permutation(len(self.P_train)))
            train_data = lgb.Dataset(self.P_train, label=self.y_train, categorical_feature=cat_feat)
            valid_data = lgb.Dataset(self.P_val, label=self.y_val)
            model = lgb.train(self.lgb_params, train_data, valid_sets=[valid_data], verbose_eval=200, )
            self.model.append(model)
            # print(model.dump_model()["tree_info"])

        print('_' * 100, '\n')

    def save(self, model_name):
        """
        Save model to folder.
        """
        super().save(model_name)
        model_name = '../models/' + model_name
        if self.model is not None:
            # self.model.save_model(model_name + '/model.h5')
            for i, mod in enumerate(self.model):
                mod.save_model(f'{model_name}/tree/model{i}.h5')
        np.save(model_name + '/P_train.npy', self.P_train)
        np.save(model_name + '/P_test.npy', self.P_test)

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """
        super().load(model_name=model_name, fast=fast)
        model_name = '../models/' + model_name
        # self.model = tf.keras.models.load_model(model_name + '/model.h5')
        self.model = []
        for i in range(self.n_estimators):
            self.model.append(lgb.Booster(model_file=f'{model_name}/tree/model{i}.h5'))

        if not fast:
            self.P_train = np.load(model_name + '/P_train.npy')
            self.P_test = np.load(model_name + '/P_test.npy')
