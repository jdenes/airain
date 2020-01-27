import os, requests, json
import datetime
from datetime import date
import pandas as pd
import numpy as np
import glob
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import tensorflow as tf


def fetchCryptoRate(from_currency, to_currency, start, end, freq):
	
	base_url = "https://poloniex.com/public?command=returnChartData" + "&currencyPair=" + from_currency + "_" + to_currency
	
	start, end = datetime.strptime(start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
	tmp1 = start
	tmp2 = start + timedelta(weeks=12)
	while tmp2 <= end:
		x1, x2 = datetime.timestamp(tmp1), datetime.timestamp(tmp2)
		main_url = base_url + "&start=" + str(x1) + "&end=" + str(x2) + "&period=" + str(freq*60)
		print('Fetching:', main_url)
		req_ob = requests.get(main_url) 
		result = req_ob.json()
		
		with open('data/raw/crypto/' + tmp1.strftime("%Y%m%d") + '-' + tmp2.strftime("%Y%m%d") + '-' + from_currency + to_currency + str(freq) + '.json', 'w') as outfile:
			json.dump(result, outfile)
		
		tmp1, tmp2 = tmp1 + timedelta(weeks=12), tmp2 + timedelta(weeks=12)
		if tmp1 < end and tmp2 > end: tmp2 = end
	

def fetchExchangeRate(from_currency, to_currency, freq, api_key) : 

	base_url = r"https://www.alphavantage.co/query?function=FX_INTRADAY"
	main_url = base_url + "&from_symbol=" + from_currency + "&to_symbol=" + to_currency
	main_url = main_url + "&interval=" + str(freq) + "min&outputsize=full" + "&apikey=" + api_key
	print('Fetching:', main_url)
	req_ob = requests.get(main_url) 
	result = req_ob.json() 

	with open('data/raw/' + str(date.today()) + '-' + from_currency + to_currency + str(freq) + '.json', 'w') as outfile:
		json.dump(result, outfile)


def structure_cryptodata():

	df = pd.DataFrame()
	for f in glob.glob('./data/raw/crypto/*-*-USDCBTC5.json'):
		with open(f) as json_file:
			data = json.load(json_file)
			df1 = pd.DataFrame.from_dict(data).set_index('date')
			df1.index = pd.to_datetime(df1.index, unit='s')
			df = df.combine_first(df1)
	df.to_csv('./data/csv/dataset_crypto.csv', encoding='utf-8')


def structure_data():

	dfA = pd.DataFrame()
	for f in glob.glob('./data/raw/2020-*-*-EURGBP5.json'):
		with open(f) as json_file:
			data = json.load(json_file)
			df1 = pd.DataFrame.from_dict(data["Time Series FX (5min)"], orient='index')
			dfA = dfA.combine_first(df1)
	dfA.columns = [('EURGBP ' + x) for x in dfA.columns]

	dfB = pd.DataFrame()
	for f in glob.glob('./data/raw/2020-*-*-GBPEUR5.json'):
		with open(f) as json_file:
			data = json.load(json_file)
			df1 = pd.DataFrame.from_dict(data["Time Series FX (5min)"], orient='index')
			dfB = dfB.combine_first(df1)
	dfB.columns = [('GBPEUR ' + x) for x in dfB.columns]

	if not dfA.index.equals(dfB.index):
		print('Warning: index not aligned btw EUR->GBP and GPB->EUR')
	df = pd.concat([dfA, dfB], axis=1, sort=True).dropna()
	
	df.to_csv('./data/csv/dataset.csv', encoding='utf-8')


def simple_prediction(h=10, data='crypto', mode='now'):

	if data == 'crypto':
		df = pd.read_csv('./data/csv/dataset_crypto.csv', encoding='utf-8', index_col=0)
		saved_label = (df['weightedAverage'].shift(-1) >= df['weightedAverage']).astype(int)
	else:
		if mode == 'now':
			df = pd.read_csv('./data/csv/dataset.csv', encoding='utf-8', index_col=0)
		else:
			df = pd.read_csv('./data/csv/dataset_long.csv', encoding='utf-8', index_col=0)
		saved_label = (df['EURGBP 1. open'] >= df['EURGBP 4. close']).astype(int).shift(-1)
	df, saved_label = df[pd.notnull(saved_label)], saved_label[pd.notnull(saved_label)]
	print("Available data shape:", df.shape)
	
	to_add = pd.DataFrame()
	for i in range(1,h):
		shifted_df = df.shift(i)
		to_add = pd.concat([to_add, shifted_df], axis=1, sort=True)
	df = pd.concat([df, to_add], axis=1, sort=True)
	df['goes_up'] = saved_label
	df = df.dropna()

	X = df.loc[:,df.columns!='goes_up'].to_numpy()
	y = df['goes_up'].to_numpy()
	print("X and y data shape:", X.shape, y.shape)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
	
	# clf = RandomForestClassifier(n_estimators=1000)
	clf = svm.SVC(gamma='scale', kernel='rbf')
	# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 1000, 500, 50))
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	
	print("Baseline: average 'goes_up' value:", y_train.mean())
	return accuracy_score(y_test, y_pred)


def hard_prediction(h=10, data='crypto', mode='now', gpu=True):
	
	if not gpu: os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
	
	if data == 'crypto':
		df = pd.read_csv('./data/csv/dataset_crypto.csv', encoding='utf-8', index_col=0)
		saved_label = (df['weightedAverage'].shift(-1) - df['weightedAverage'])#.apply(lambda x: np.sign(x) if pd.notnull(x) else x)
	else:
		if mode =='now':
			df = pd.read_csv('./data/csv/dataset.csv', encoding='utf-8', index_col=0)	
		else:
			df = pd.read_csv('./data/csv/dataset_long.csv', encoding='utf-8', index_col=0)
			df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
			idx = list(pd.date_range(df.index.min(), df.index.max()))
			df = df.reindex(idx, method='ffill')
			df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
		saved_label = (df['EURGBP 1. open'] >= df['EURGBP 4. close']).astype(int).shift(-1)
	df, saved_label = df[pd.notnull(saved_label)], saved_label[pd.notnull(saved_label)]
	print("Available data shape:", df.shape)
	
	data, labels = [], []
	for i, row in df.iterrows():
		end = datetime.strptime(i, '%Y-%m-%d %H:%M:%S')
		if mode == 'now':
			ind = [str(end - timedelta(minutes=x*5)) for x in range(h)]
		else:
			ind = [str(end - timedelta(days=x)) for x in range(h)]
		if all(x in df.index for x in ind):
			slicing = df.loc[ind]
			data.append(np.array(slicing))
			labels.append(saved_label[i])

	X = np.array(data)
	y = np.array(labels)
	# y = y.reshape((len(y), 1))
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
	x_mean, x_max = X_train.mean(axis=0), X_train.max(axis=0)
	X_train, X_test = (X_train - x_mean) / x_max, (X_test - x_mean) / x_max

	print("X and y data shape:", X.shape, y.shape)
	print(list(y_train))

	BATCH_SIZE = 1000
	BUFFER_SIZE = 100000
	EPOCHS = 1
	STEPS = 2000
	VALSTEPS = 500
	
	train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
	val_data = val_data.batch(BATCH_SIZE).repeat()

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.LSTM(50, input_shape=X.shape[-2:]))
	model.add(tf.keras.layers.Dense(1))
	model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mse')
	
	# model = tf.keras.models.Sequential()
	# model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=X.shape[-2:]))
	# model.add(tf.keras.layers.RepeatVector(1))
	# model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))
	# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu')))
	# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
	# model.compile(loss='mse', optimizer='adam')
	
	history = model.fit(train_data, epochs=EPOCHS,
									steps_per_epoch=STEPS,
									validation_steps=VALSTEPS,
									validation_data=val_data)

	y_pred = model.predict(X_test)
	np.save('data/essai.npy', y_pred)
	y_test = y_test.flatten()
	print(y_test, y_pred)
	print("Baseline: average 'goes_up' value:", y_test.mean())
	# return np.equal(y_test, y_pred).mean()

def back_test(X, y):

	pass


if __name__ == "__main__" : 
	
	freq = 5
	end = '2020-01-27 00:00:00'
	start = '2019-01-27 00:00:00'
	h = 30
	api_key = "H2T4H92C43D9DT3D"
	
	if False:
		for from_currency, to_currency in [('EUR', 'GBP'), ('GBP', 'EUR')]:
			fetchExchangeRate(from_currency, to_currency, freq, api_key)
		structure_data()
	
	if False:
		fetchCryptoRate('USDC', 'BTC', start, end, freq)
		structure_cryptodata()

	
	score = hard_prediction(h=h, data='crypto', mode='long', gpu=True)
	# print(score)
	# score = simple_prediction(h=h, data='crypto', mode='now')
	print("Final model score:", score)
	