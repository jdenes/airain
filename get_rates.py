import os, requests, json
import datetime
from datetime import date
import pandas as pd
import numpy as np
import glob
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import load_data, fetch_crypto_rate, fetch_exchange_rate, structure_crypto, structure_currencies, compute_metrics


def simple_prediction(h=10, data='crypto', mode='now'):

	df, labels = load_data(datatype=data)
	
	y_min, y_max = labels.min(), labels.max()
	df = 2*(df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0)) - 1
	labels = 2*(labels - y_min) / (y_max - y_min) - 1

	print("Available data shape:", df.shape)
	
	to_add = pd.DataFrame()
	for i in range(1,h):
		shifted_df = df.shift(i)
		to_add = pd.concat([to_add, shifted_df], axis=1, sort=True)
	df = pd.concat([df, to_add], axis=1, sort=True)
	df['labels'] = labels
	df = df.dropna()

	X = df.loc[:,df.columns!='labels'].to_numpy()
	y = df['labels'].to_numpy()
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
	print("Train:", X_train.shape, "Test:", X_test.shape)
		
	model = RandomForestRegressor(n_estimators=10)
	# model = svm.SVR(gamma='scale', kernel='rbf')
	# model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 1000, 500, 50))
	model.fit(X_train, y_train)
	
	y_pred = model.predict(X_test).flatten()
	y_trans = (y_pred + 1)*(y_max - y_min)/2 + y_min
	y_comp = (y_test + 1)*(y_max - y_min)/2 + y_min

	plt.plot((y_trans-y_comp)/y_comp, '.')
	plt.show()

	plt.plot(y_comp, y_trans, '.')
	plt.show()

	return compute_metrics(y_comp, y_trans)

def hard_prediction(h=10, data='crypto', mode='now', freq=5, gpu=True):
	
	if not gpu: os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
	
	df, labels = load_data(datatype=data)

	y_min, y_max = labels.min(), labels.max()
	df = 2*(df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0)) - 1
	labels = 2*(labels - y_min) / (y_max - y_min) - 1
	
	print("Available data shape:", df.shape)
	# print(df.describe())
	
	from tqdm import tqdm
	X, y = [], []
	for i, row in tqdm(df.iterrows()):
		end = datetime.strptime(i, '%Y-%m-%d %H:%M:%S')
		if mode == 'now':
			ind = [str(end - timedelta(minutes=x*freq)) for x in range(h)]
		else:
			ind = [str(end - timedelta(days=x)) for x in range(h)]
		if all(x in df.index for x in ind):
			slicing = df.loc[ind]
			X.append(np.array(slicing))
			y.append(labels[i])
	X = np.array(X)
	y = np.array(y)
	
	# y = y.reshape((len(y), 1))
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)
	print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

	BATCH_SIZE = 1000
	BUFFER_SIZE = 100000
	EPOCHS = 20
	STEPS = 500
	VALSTEPS = 50
	
	train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
	val_data = val_data.batch(BATCH_SIZE).repeat()

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.LSTM(50, input_shape=X.shape[-2:], return_sequences=True))
	model.add(tf.keras.layers.LSTM(16, activation='tanh'))
	model.add(tf.keras.layers.Dense(1))
	model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
	
	# model = tf.keras.models.Sequential()
	# model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=X.shape[-2:]))
	# model.add(tf.keras.layers.RepeatVector(1))
	# model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))
	# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu')))
	# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
	# model.compile(loss='mse', optimizer='adam')
	
	print(model.summary())

	history = model.fit(train_data, epochs=EPOCHS,
									steps_per_epoch=STEPS,
									validation_steps=VALSTEPS,
									validation_data=val_data)

	y_pred = model.predict(X_test).flatten()
	y_trans = (y_pred + 1)*(y_max - y_min)/2 + y_min
	y_comp = (y_test + 1)*(y_max - y_min)/2 + y_min

	plt.plot((y_trans-y_comp)/y_comp, '.')
	plt.show()

	plt.plot(y_comp, y_trans, '.')
	plt.show()

	return compute_metrics(y_comp, y_trans)


if __name__ == "__main__" : 
	
	freq = 5
	end = '2020-01-27 00:00:00'
	start = '2018-07-10 00:00:00'
	h = 20
	api_key = "H2T4H92C43D9DT3D"
	from_curr = 'USDC'
	to_curr = 'BTC'
	
	if False:
		for x, y in [('EUR', 'GBP'), ('GBP', 'EUR')]:
			fetch_exchange_rate(x, y, freq, api_key)
		structure_currencies(from_curr, to_curr, freq)
	
	if False:
		fetch_crypto_rate(from_curr, to_curr, start, end, freq)
		structure_crypto(from_curr, to_curr, freq)

	
	score = hard_prediction(h=h, data='crypto', mode='now', freq=freq, gpu=True)
	score = simple_prediction(h=h, data='crypto', mode='now')
	print("Final model scores:", score)
	