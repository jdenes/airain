import os
import json
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import lightgbm as lgb
import tensorflow as tf

from utils.basics import omega2assets, evaluate_portfolio, normalize_data
from utils.metrics import classification_perf
from utils.plots import nice_plot

logger = logging.getLogger(__name__)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

####################################################################################


class Trader(object):
    """
    A general trader-forecaster to inherit from.
    """

    def __init__(self, h=10, seed=123, forecast=1, normalize=False,
                 t0='2000-01-01', t1='2019-01-01', t2='2020-01-01',
                 load_from=None, fast_load=True):
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
        self.valsize = None
        self.X_train = None
        self.P_train = None
        self.y_train = None
        self.ind_train = None
        self.X_test = None
        self.P_test = None
        self.y_test = None
        self.ind_test = None
        self.X_val = None
        self.P_val = None
        self.y_val = None
        self.ind_val = None

        self.t0, self.t1, self.t2 = t0, t1, t2

    def transform_data(self, df, labels, keep_last=False):
        """
        Converts dataframe file to appropriate data format for this agent type.
        """
        return None, None, None, np.array(0)

    def ingest_traindata(self, df, labels, duplicate=False, testsize=0.1, valsize=0.1):
        """
        Loads data from csv file depending on data type.
        """

        self.testsize = testsize
        self.valsize = valsize

        train_ind = (df.index >= self.t0) & (df.index < self.t1)
        val_ind = (df.index >= self.t1) & (df.index < self.t2)
        test_ind = (df.index >= self.t2)

        df_train, labels_train = df[train_ind], labels[train_ind]
        if duplicate:
            df_train, labels_train = pd.concat([df_train] * 10, axis=0), pd.concat([labels_train] * 10, axis=0)
        self.x_max, self.x_min = df_train.max(axis=0), df_train.min(axis=0)
        self.p_max, self.p_min = self.x_max, self.x_min
        self.y_min, self.y_max = labels_train.min(), labels_train.max()

        X, P, y, ind = self.transform_data(df_train, labels_train)
        self.X_train, self.P_train, self.y_train, self.ind_train = X, P, y, ind

        df_test, labels_test = df[test_ind], labels[test_ind]
        X, P, y, ind = self.transform_data(df_test, labels_test)
        self.X_test, self.P_test, self.y_test, self.ind_test = X, P, y, ind
        del df_test, labels_test

        df_val, labels_val = df[val_ind], labels[val_ind]
        X, P, y, ind = self.transform_data(df_val, labels_val)
        self.X_val, self.P_val, self.y_val, self.ind_val = X, P, y, ind
        del df_val, labels_val

    def train(self):
        """
        Using prepared data, trains model depending on agent type.
        """
        print('_' * 100, '\n')
        print(f'Training on {self.P_train.shape[0]} examples, with {self.P_train.shape[-1]} features...')
        y_mean = (self.y_train > 0).mean()
        print(f'Baseline for change accuracy is {round(max(y_mean, 1 - y_mean), 4)}.')
        print('_' * 100, '\n')

    def test(self, plot=False):
        """
        Once model is trained, uses test data to output performance metrics.
        """
        y_train, y_val, y_test = self.y_train, self.y_val, self.y_test

        print("Performance metrics on train:")
        y_pred = self.predict(self.X_train, self.P_train)
        print(classification_perf(y_train, y_pred))

        print("Performance metrics on val:")
        y_pred = self.predict(self.X_val, self.P_val)
        print(classification_perf(y_val, y_pred))

        print("Performance metrics on test:")
        y_pred = self.predict(self.X_test, self.P_test)
        print(classification_perf(y_test, y_pred))

    def predict(self, X, P):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        pass

    def save(self, model_name):
        """
        Save model to folder.
        """

        model_name = '../models/' + model_name
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        to_rm = ['model', 'optimizer', 'x_max', 'x_min', 'p_min', 'p_max',
                 'X_train', 'P_train', 'y_train', 'ind_train',
                 'X_test', 'P_test', 'y_test', 'ind_test',
                 'X_val', 'P_val', 'y_val', 'ind_val']
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
        np.save(model_name + '/y_train.npy', self.y_train)
        np.save(model_name + '/ind_train.npy', self.ind_train)
        np.save(model_name + '/X_val.npy', self.X_val)
        np.save(model_name + '/y_val.npy', self.y_val)
        np.save(model_name + '/ind_val.npy', self.ind_val)
        np.save(model_name + '/X_test.npy', self.X_test)
        np.save(model_name + '/y_test.npy', self.y_test)
        np.save(model_name + '/ind_test.npy', self.ind_test)

        if isinstance(self.P_train, pd.DataFrame):
            self.P_train.to_csv(model_name + '/P_train.csv')
            self.P_test.to_csv(model_name + '/P_test.csv')
            self.P_val.to_csv(model_name + '/P_val.csv')
        else:
            np.save(model_name + '/P_train.npy', self.P_train)
            np.save(model_name + '/P_test.npy', self.P_test)
            np.save(model_name + '/P_val.npy', self.P_val)

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
            self.y_train = np.load(model_name + '/y_train.npy')
            self.ind_train = np.load(model_name + '/ind_train.npy')
            self.X_val = np.load(model_name + '/X_val.npy')
            self.y_val = np.load(model_name + '/y_val.npy')
            self.ind_val = np.load(model_name + '/ind_val.npy')
            self.X_test = np.load(model_name + '/X_test.npy')
            self.y_test = np.load(model_name + '/y_test.npy')
            self.ind_test = np.load(model_name + '/ind_test.npy')
            try:
                self.P_train = pd.read_csv(model_name + '/P_train.csv')
            except FileNotFoundError:
                self.P_train = np.load(model_name + '/P_train.npy')
            try:
                self.P_val = pd.read_csv(model_name + '/P_val.csv')
            except FileNotFoundError:
                self.P_val = np.load(model_name + '/P_val.npy')
            try:
                self.P_test = pd.read_csv(model_name + '/P_test.csv')
            except FileNotFoundError:
                self.P_test = np.load(model_name + '/P_test.npy')


####################################################################################


class LGBMTrader(Trader):
    """
    A trader-forecaster based on a LightGBM model.
    """

    def __init__(self, h=10, seed=123, forecast=1,
                 t0='2000-01-01', t1='2019-01-01', t2='2020-01-01',
                 normalize=False, load_from=None, fast_load=True):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize,
                         t0=t0, t1=t1, t2=t2,
                         load_from=load_from, fast_load=fast_load)

        self.lgb_params = {
            'objective': 'binary',  # multiclass
            # 'num_class': 3,
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',  # dart, goss
            'subsample': 1.0,  # 0.5
            'subsample_freq': 1,
            'learning_rate': 0.01,  # 0.03
            'top_rate': 0.7,
            'other_rate': 0.1,
            'num_leaves': 2 ** 12 - 1,
            'min_data_in_leaf': 2 ** 12 - 1,
            'feature_fraction': 0.7,  # between 0.4 and 0.6
            'max_bin': 11,  # 255
            'num_iterations': 10000,
            'boost_from_average': True,
            'verbose': -1,
            # 'early_stopping_rounds': 1000,
        }

        if load_from is not None:
            self.load(model_name=load_from, fast=fast_load)

    def transform_data(self, df, labels, keep_last=False):
        """
        Given data and labels, transforms it into suitable format and return them.
        """

        ind = np.array(df.index.to_list())
        df, labels = df.reset_index(drop=True), labels.reset_index(drop=True)
        if self.normalize:
            df = normalize_data(df, self.x_max, self.x_min)
            # labels = normalize_data(labels, self.y_max, self.y_min)

        X, P, y = np.array([]), df, labels.to_numpy()
        return X, P, y, ind

    def train(self, batch_size=80, buffer_size=10000, epochs=50, steps=1000, valsteps=100, gpu=True):
        """
        Using prepared data, trains model depending on agent type.
        """

        super().train()

        cat_feat = ['asset']
        # idx = (np.random.permutation(len(self.P_train)))
        train_data = lgb.Dataset(self.P_train, label=self.y_train, categorical_feature=cat_feat)
        valid_data = lgb.Dataset(self.P_val, label=self.y_val)
        self.model = lgb.train(self.lgb_params, train_data, valid_sets=[valid_data], verbose_eval=200, )

        # print(model.dump_model()["tree_info"])

        print('_' * 100, '\n')

    def test(self, plot=False):
        """
        Once model is trained, uses test data to output performance metrics.
        """

        super().test()

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(14, 14))
            lgb.plot_importance(self.model, ax=ax[0], max_num_features=25, precision=0,
                                importance_type='split', title='In number of splits')
            lgb.plot_importance(self.model, ax=ax[1], max_num_features=25, precision=0,
                                importance_type='gain', title='In total splits gain')
            fig.tight_layout()
            plt.show()

            # y_pred = self.predict(self.X_test, self.P_test)
            # i = self.y_test != 0
            # plt.plot((y_pred[i] - self.y_test[i]), '.')
            # plt.show()
            # plt.plot(self.y_test, y_pred, '.')
            # plt.show()

    def predict(self, X, P):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        y_pred = self.model.predict(P)
        y_pred = y_pred.round(0).astype(int)
        return y_pred

    def save(self, model_name):
        """
        Save model to folder.
        """
        super().save(model_name)
        model_name = '../models/' + model_name
        if self.model is not None:
            self.model.save_model(f'{model_name}/model.h5')

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """
        super().load(model_name=model_name, fast=fast)
        model_name = '../models/' + model_name
        self.model = lgb.Booster(model_file=f'{model_name}/model.h5')


####################################################################################


class LstmTrader(Trader):
    """
    A trader-forecaster based on a LSTM neural network.
    """

    def __init__(self, h=10, seed=123, forecast=1,
                 t0='2000-01-01', t1='2019-01-01', t2='2020-01-01',
                 normalize=False, load_from=None, fast_load=True):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize,
                         t0=t0, t1=t1, t2=t2,
                         load_from=load_from, fast_load=fast_load)

        self.batch_size = None
        self.buffer_size = None
        self.epochs = None
        self.steps = None
        self.valsteps = None
        self.gpu = None

        if load_from is not None:
            self.load(model_name=load_from, fast=fast_load)

    def transform_data(self, df, labels, keep_last=False):
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

        for i in range(self.h - 1, len(df)):  # range(len(df)):
            indx = [int(i - self.h + x + 1) for x in range(self.h)]
            X.append(df[indx])
            P.append(df[i])
            y.append(labels[i])
            ind.append(index[i])

        X, P, y, ind = np.array(X), np.array(P), np.array(y), np.array(ind)
        P = pd.DataFrame(P, columns=columns)
        return X, P, y, ind

    def train(self, batch_size=80, buffer_size=10000, epochs=50, steps=1000, valsteps=1000, gpu=True):
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

        super().train()

        train_data = tf.data.Dataset.from_tensor_slices(
            ({'input_X': self.X_train, 'input_P': self.P_train}, self.y_train))
        train_data = train_data.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        valid_data = tf.data.Dataset.from_tensor_slices(
            ({'input_X': self.X_val, 'input_P': self.P_val}, self.y_val))
        valid_data = valid_data.shuffle(self.buffer_size).batch(self.batch_size).repeat()
        initializer = tf.keras.initializers.GlorotNormal()

        input_layer = tf.keras.layers.Input(shape=self.X_train.shape[-2:], name='input_X')
        price_layer = tf.keras.layers.Input(shape=self.P_train.shape[-1], name='input_P')
        lstm_layer = tf.keras.layers.LSTM(64, kernel_initializer=initializer, name='lstm')(input_layer)
        output_layer = tf.keras.layers.Dense(2, kernel_initializer=initializer, name='output')(lstm_layer)
        self.model = tf.keras.Model(inputs=[input_layer, price_layer], outputs=output_layer)
        self.model.summary()

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='../models/checkpoint.hdf5',
                                                        monitor='val_accuracy',
                                                        save_best_only=True)
        self.model.fit(train_data,
                       epochs=self.epochs,
                       steps_per_epoch=self.steps,
                       validation_steps=self.valsteps,
                       validation_data=valid_data,
                       callbacks=[checkpoint]
                       )

        print('_' * 100, '\n')

    def predict(self, X, P):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        y_pred = self.model.predict((X, P))
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def test(self, plot=False):
        """
        Once model is trained, uses test data to output performance metrics.
        """
        super().test()

    def save(self, model_name):
        """
        Save model to folder.
        """
        super().save(model_name)
        model_name = '../models/' + model_name
        if self.model is not None:
            tf.keras.models.save_model(self.model, f'{model_name}/model.h5')

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """
        super().load(model_name=model_name, fast=fast)
        model_name = '../models/' + model_name
        self.model = tf.keras.models.load_model(f'{model_name}/model.h5')


####################################################################################


class LstmContextTrader(Trader):
    """
    A trader-forecaster based on a LSTM neural network.
    """

    def __init__(self, h=10, seed=123, forecast=1,
                 t0='2000-01-01', t1='2019-01-01', t2='2020-01-01',
                 normalize=False, load_from=None, fast_load=True):
        """
        Initialize method.
        """

        super().__init__(h=h, seed=seed, forecast=forecast, normalize=normalize,
                         t0=t0, t1=t1, t2=t2,
                         load_from=load_from, fast_load=fast_load)

        tf.random.set_seed(self.seed)
        self.batch_size = None
        self.buffer_size = None
        self.epochs = None
        self.steps = None
        self.valsteps = None
        self.num_assets = None
        self.gpu = None
        self.model = None
        self.optimizer = None

        if load_from is not None:
            self.load(model_name=load_from, fast=fast_load)

    def transform_data(self, df, labels, keep_last=False):
        """
        Given data and labels, transforms it into suitable format and return them.
        """

        if self.normalize:
            norm_df = normalize_data(df, self.x_max, self.x_min)
        else:
            norm_df = df

        norm_df = norm_df[[c for c in norm_df if not ('mean' in c or 'std' in c)]]
        dates = sorted(norm_df.index.unique().to_list())
        assets = sorted(norm_df['asset'].unique())
        self.num_assets = norm_df['asset'].nunique()
        assets_df = [norm_df[norm_df.asset == asset].sort_index().drop('asset', 1) for asset in assets]

        X, P, y, ind = [], [], [], []
        count = 0

        from tqdm import tqdm
        for i in tqdm(range(self.h - 1, len(dates))):
            day = dates[i]
            tmp = []
            for j, asset in enumerate(assets):
                asset_df = assets_df[j]
                asset_df = asset_df[asset_df.index <= day].tail(self.h).to_numpy(dtype=np.float32)
                tmp.append(asset_df)
            if not any(x.shape[0] != self.h for x in tmp):
                X.append(tmp)
                P.append(df[df.index == day].to_numpy(dtype=np.float32))
                y.append(labels[labels.index == day].to_numpy(dtype=np.float32))
                ind.append(day)
            else:
                count += 1

        X, P, y, ind = np.array(X), np.array(P), np.array(y), np.array(ind)
        return X, P, y, ind

    def init_model(self):
        """
        Create TensorFlow internal model.
        """
        initializer = tf.keras.initializers.GlorotNormal()
        # Input layer shape is (batch, assets, window, features)
        input_layer = tf.keras.layers.Input(shape=self.X_train.shape[-3:], name='input_X')
        price_layer = tf.keras.layers.Input(shape=self.P_train.shape[-2:], name='input_P')
        # Step 1: one LSTM per feature, taking an (asset, window) matrix as input
        lstm_layers = []
        for i in range(self.X_train.shape[-1]):
            feature_tensor = input_layer[:, :, :, i]
            lstm_layer = tf.keras.layers.SimpleRNN(1, kernel_initializer=initializer, name=f'lstm_{i}')(feature_tensor)
            lstm_layers.append(lstm_layer)
        # Step 2: one dense layer per asset+cash to discuss independently about LSTM result, output in 1 dim
        # conv = []
        # fusion = tf.stack(lstm_layers, axis=1)
        # for i in range(fusion.shape[-1]):
        #     asset_slice = fusion[:, :, i]
        #     asset_dense = tf.keras.layers.Dense(1)(asset_slice)
        #     conv.append(asset_dense)
        # conv = tf.concat(conv, axis=1)
        conv = tf.concat(lstm_layers, axis=1)
        conv = tf.keras.layers.Dense(self.num_assets+1)(conv)
        # Step 5: vote using softmax
        output_layer = tf.keras.layers.Softmax(name='output')(conv)
        self.model = tf.keras.Model(inputs=[input_layer, price_layer], outputs=output_layer)
        self.model.summary()

    def loss(self, features, future_prices):
        """
        Loss of the model is defined as minus the value of the portfolio (nb of asset times asset price).

        :param features:
        :param future_prices:
        :return:
        """
        portfolio = self.model(features)
        loss_value = -tf.math.reduce_mean(tf.math.reduce_sum(portfolio * future_prices, axis=1))
        # entropy = tf.math.reduce_mean(-tf.math.reduce_sum(portfolio * tf.math.log(portfolio), axis=1))
        return loss_value  # - 9e-4 * entropy

    def gradient(self, features, future_prices):
        """
        Gradient definition or TF gradient descent.

        :param features:
        :param future_prices:
        :return:
        """
        with tf.GradientTape() as tape:
            loss_value = self.loss(features, future_prices)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, batch_size=264, buffer_size=10000, epochs=10, patience=20, gpu=True):
        """
        Using prepared data, trains model depending on agent type.
        """

        self.gpu = gpu
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs

        if not self.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        super().train()

        train_data = tf.data.Dataset.from_tensor_slices(
            ({'input_X': self.X_train, 'input_P': self.P_train}, self.y_train))
        train_data = train_data.cache().shuffle(self.buffer_size).batch(self.batch_size)
        valid_data = tf.data.Dataset.from_tensor_slices(
            ({'input_X': self.X_val, 'input_P': self.P_val}, self.y_val))
        valid_data = valid_data.shuffle(self.buffer_size).batch(self.batch_size)

        self.init_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        train_loss_results = []
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        best_loss, best_epoch, best_model = 1000, self.epochs, None
        no_progress = 0

        @tf.function
        def train_step(X, y):
            cash_price = tf.ones((y.shape[0], 1))  # for cash, price change is always zero
            future_prices = tf.concat((cash_price, y), axis=1)
            loss_value, grads = self.gradient(X, future_prices)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            return loss_value

        @tf.function
        def val_step(X, y):
            cash_price = tf.zeros((y.shape[0], 1))
            future_prices = tf.concat((cash_price, y), axis=1)
            loss_value = self.loss(X, future_prices)
            epoch_val_loss_avg.update_state(loss_value)
            return loss_value

        for epoch in range(self.epochs):

            for features, labels in train_data:
                train_step(features, labels)

            for val_features, val_labels in valid_data:
                val_step(val_features, val_labels)

            train_loss = epoch_loss_avg.result()
            val_loss = epoch_val_loss_avg.result()
            train_loss_results.append(train_loss)
            print(f"Epoch {epoch+1:03d}/{self.epochs} - loss: {train_loss:.5f} - val_loss: {val_loss:.5f}")

            if val_loss < best_loss:
                no_progress = 0
                best_loss, best_epoch = val_loss, epoch
                tf.keras.models.save_model(self.model, '../models/best.h5', include_optimizer=True)
            else:
                no_progress += 1

            epoch_loss_avg.reset_states()
            epoch_val_loss_avg.reset_states()

            if no_progress > patience:
                print(f'No progress since {patience} epochs, early stopping')
                break

        if best_epoch != self.epochs-1:
            print(f'Restoring best model: val_loss was {best_loss:.5f} at epoch {best_epoch+1:03d}.')
            self.model = tf.keras.models.load_model('../models/best.h5', compile=False)

        print('_' * 100, '\n')
        print(self.model(features)[-1].numpy())
        print(self.model(features)[0].numpy())
        print(np.argmax(self.model(features).numpy(), axis=1))

    def predict(self, X, P):
        """
        Once the model is trained, predicts output if given appropriate (transformed) data.
        """
        return self.model((X, P))

    def test(self, plot=False):
        """
        Once model is trained, uses test data to output performance metrics.
        """

        # Prediction at day t is computed at day t-1
        X, P, ind = self.X_test, self.P_test, self.ind_test
        omegas = self.model((X, P))
        gamble = balance = ref_balance = 100000
        history, ref_history, index = [balance], [ref_balance], [ind[0]]

        for i in range(1, len(ind)):
            today = ind[i]
            open_price, close_price = P[i][:, 3], P[i][:, 2]
            open_price, close_price = np.concatenate(([1.0], open_price)), np.concatenate(([1.0], close_price))
            omega = omegas[i - 1]
            ref_omega = np.ones((len(omega))) / len(omega)

            # Omega gives me proportions, I need to make a number of assets out of that
            portfolio = omega2assets(gamble, omega, open_price)
            ref_portfolio = omega2assets(gamble, ref_omega, open_price)
            morning_value = evaluate_portfolio(portfolio, open_price)
            ref_morning_value = evaluate_portfolio(ref_portfolio, open_price)

            # Then days goes on and portfolio value changes
            evening_value = evaluate_portfolio(portfolio, close_price)
            ref_evening_value = evaluate_portfolio(ref_portfolio, close_price)
            balance += (evening_value - morning_value)
            ref_balance += (ref_evening_value - ref_morning_value)
            history.append(balance), ref_history.append(ref_balance), index.append(today)

        nice_plot(index, [history, ref_history], ['Portfolio', 'Benchmark'], title=f'Portfolio balance evolution')

    def save(self, model_name):
        """
        Save model to folder.
        """
        super().save(model_name)
        model_name = '../models/' + model_name
        if self.model is not None:
            tf.keras.models.save_model(self.model, f'{model_name}/model.h5', include_optimizer=True)

    def load(self, model_name, fast=False):
        """
        Load model from folder.
        """
        super().load(model_name=model_name, fast=fast)
        model_name = '../models/' + model_name
        self.model = tf.keras.models.load_model(f'{model_name}/model.h5', compile=False)
