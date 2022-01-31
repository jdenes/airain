import numpy as np
import tensorflow as tf


class Autoencoder(tf.keras.models.Model):

    def __init__(self, latent_dim, width, height):

        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.width = width
        self.height = height
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.width * self.height, activation='sigmoid'),
            tf.keras.layers.Reshape((self.width, self.height))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

x_train = np.array([item for sublist in trader.X_train for item in sublist])
x_test = np.array([item for sublist in trader.X_test for item in sublist])
y_train = np.array([int(item > 1) for sublist in trader.y_train for item in sublist])
y_test = np.array([int(item > 1) for sublist in trader.y_test for item in sublist])
autoencoder = Autoencoder(latent_dim=64, width=x_test.shape[1], height=x_test.shape[2])
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
autoencoder.fit(x_train, x_train, epochs=20, shuffle=True, validation_data=(x_test, x_test))

encoded_x_train = autoencoder.encoder(x_train).numpy()
encoded_x_test = autoencoder.encoder(x_test).numpy()
from sklearn.svm import SVC
from sklearn.metrics import classification_report

svc = SVC(kernel='rbf', gamma='scale')
svc.fit(encoded_x_train, y_train)
y_pred = svc.predict(encoded_x_train)
print(classification_report(y_train, y_pred))