from utils import load_data, fetch_crypto_rate, fetch_exchange_rate, structure_crypto, structure_currencies
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, r2_score

class Trader(object):

	def __init__(self, freq=5, h=10, seed=123, forecast=1):
		self.freq = freq
		self.h = h
		self.forecast = forecast
		self.seed = seed
		self.model = None
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.X_val = None
		self.y_val = None

	def transform_data(self, data):
		"""
		Converts .csv file to appropriate data format for training for this agent type.
		"""
		pass
	
	def ingest_data(self, csv_file):
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
		


class LstmTrader(Trader):

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
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.X_val = None
		self.y_val = None
		self.x_max, self.x_min = None, None
		self.y_max, self.y_min = None, None
	
	def transform_data(self, df, labels):
		"""
		Given data and labels, transforms it into suitable format and return them.
		"""
		
		if self.normalize:
			df = 2*(df - self.x_min) / (self.x_max - self.x_min) - 1
			labels = 2*(labels - self.y_min) / (self.y_max - self.y_min) - 1
		
		X, y = [], []
		for i, row in tqdm(df.iterrows()):
			end = datetime.strptime(i, '%Y-%m-%d %H:%M:%S')
			if self.datatype == 'short_currency':
				ind = [str(end - timedelta(minutes=x*freq)) for x in range(h)]
			else:
				ind = [str(end - timedelta(days=x)) for x in range(h)]
			if all(x in df.index for x in ind):
				slicing = df.loc[ind]
				X.append(np.array(slicing))
				y.append(labels[i])
		
		return np.array(X), np.array(y)
	
	def ingest_traindata(self):
		"""
		Loads data from csv file depending on datatype.
		"""
		
		df, labels = load_data(datatype=self.datatype)
		self.x_max, self.x_min = df.max(axis=0), df.min(axis=0)
		self.y_min, self.y_max = labels.min(), labels.max()
		X, y = self.transform_data(df, labels)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.X_val = X_val
		self.y_val = y_val
	
	def train(self):
		"""
		Using prepared data, trains model depending on agent type.
		"""
		
		BATCH_SIZE = 1000
		BUFFER_SIZE = 100000
		EPOCHS = 10
		STEPS = 200
		VALSTEPS = 50
		
		train_data = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
		train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

		val_data = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
		val_data = val_data.batch(BATCH_SIZE).repeat()

		self.model = tf.keras.models.Sequential()
		self.model.add(tf.keras.layers.LSTM(50, input_shape=X.shape[-2:], return_sequences=True))
		self.model.add(tf.keras.layers.LSTM(16, activation='relu'))
		self.model.add(tf.keras.layers.Dense(1))
		self.model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')	
		
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
		