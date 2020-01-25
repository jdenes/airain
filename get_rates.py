import os, requests, json
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

def fetchExchangeRate(from_currency, to_currency, freq, api_key) : 

	base_url = r"https://www.alphavantage.co/query?function=FX_INTRADAY"
	main_url = base_url + "&from_symbol=" + from_currency + "&to_symbol=" + to_currency
	main_url = main_url + "&interval=" + freq + "min&outputsize=full" + "&apikey=" + api_key
	print('Fetching:', main_url)
	req_ob = requests.get(main_url) 
	result = req_ob.json() 

	with open('data/raw/' + str(date.today()) + '-' + from_currency + to_currency + freq + '.json', 'w') as outfile:
		json.dump(result, outfile)

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


def simple_prediction(h=10, mode='now'):

	if mode == 'now':
		df = pd.read_csv('./data/csv/dataset.csv', encoding='utf-8', index_col=0)
	else:
		df = pd.read_csv('./data/csv/dataset_long.csv', encoding='utf-8', index_col=0)
	
	saved_label = (df['EURGBP 1. open'] >= df['EURGBP 4. close']).astype(int).shift(-1)

	to_add = pd.DataFrame()
	for i in range(1,h):
		shifted_df = df.shift(i)
		to_add = pd.concat([to_add, shifted_df], axis=1, sort=True)
	df = pd.concat([df, to_add], axis=1, sort=True)

	df['goes_up'] = saved_label
	df = df.dropna()

	X = df.loc[:,df.columns!='goes_up'].to_numpy()
	y = df['goes_up'].to_numpy()
	print(X.shape, y.shape)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
	
	# clf = RandomForestClassifier(n_estimators=1000)
	clf = svm.SVC(gamma='scale', kernel='poly', degree=7)
	# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 1000, 500, 50))
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	
	print(y_train.mean())
	return accuracy_score(y_test, y_pred)


def hard_prediction(h=10, mode='now'):
	
	if mode =='now':
		df = pd.read_csv('./data/csv/dataset.csv', encoding='utf-8', index_col=0)	
	else:
		df = pd.read_csv('./data/csv/dataset_long.csv', encoding='utf-8', index_col=0)
		df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
		idx = list(pd.date_range(df.index.min(), df.index.max()))
		df = df.reindex(idx, method='ffill')
		df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
	
	saved_labels = (df['EURGBP 1. open'] - df['EURGBP 4. close']).apply(np.sign).shift(-1).drop(df.index.max())
	df = df.drop(df.index.max())
	print(df.shape, saved_labels.shape)
	
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
			labels.append(saved_labels[i])

	X = np.array(data)
	y = np.array(labels)
	print(X.shape, y.shape)

	import tensorflow as tf
	BATCH_SIZE = 1000
	BUFFER_SIZE = 100000
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

	train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
	val_data = val_data.batch(BATCH_SIZE).repeat()

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.LSTM(50, input_shape=X.shape[-2:]))
	model.add(tf.keras.layers.Dense(1))
	model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mse')
	
	history = model.fit(train_data, epochs=10,
									steps_per_epoch=200,
									validation_data=val_data,
									validation_steps=50)

	y_pred = np.sign(model.predict(X_test).T[0])
	print(y_test, y_pred)
	return np.equal(y_test, y_pred).mean()

def back_test(X, y):

	pass


if __name__ == "__main__" : 
	
	freq = '5'
	h = 30
	api_key = "H2T4H92C43D9DT3D"
	
	if False:
		for from_currency, to_currency in [('EUR', 'GBP'), ('GBP', 'EUR')]:
			fetchExchangeRate(from_currency, to_currency, freq, api_key)
		structure_data()
	
	# score = hard_prediction(h=h, mode='now')
	# print(score)
	score = simple_prediction(h=h, mode='now')
	print(score)
	