import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import matplotlib.pyplot as plt
from data_preparation import load_data
from utils.constants import DJIA, DJIA_PERFORMERS
from utils.basics import normalize_data

FOLDER = '../data/yahoo/'
T0 = '2000-01-01'
T1 = '2020-01-01'
T2 = '2021-01-01'

df, labels = load_data(FOLDER, DJIA, T0, T1)
df = df[['open', 'high', 'low', 'close', 'volume', 'ratio']]
x_max, x_min = df.max(axis=0), df.min(axis=0)
df = normalize_data(df, x_max, x_min).sample(frac=1)

BATCH_SIZE = 2000
NOISE_SIZE = 10
EPOCHS = 50
train_dataset = tf.data.Dataset.from_tensor_slices(df).shuffle(10000).batch(BATCH_SIZE)


def make_generator_model(input_dim=100, n_features=98):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1500, activation='relu', input_dim=input_dim))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    model.add(tf.keras.layers.Dense(n_features, activation='relu'))
    return model


def make_discriminator_model(n_features=98):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(250, activation='relu', input_dim=n_features))
    model.add(tf.keras.layers.Dense(150, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


generator = make_generator_model(n_features=df.shape[1])
discriminator = make_discriminator_model(n_features=df.shape[1])
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images):
    noise = tf.random.normal([len(df), 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train():

    for epoch in range(EPOCHS):
        image_batch = df
        gen_loss, disc_loss = train_step(image_batch)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} - gen_loss: {gen_loss:.5f} - disc_loss: {disc_loss:.5f}")


generator.summary()
discriminator.summary()
train()

noise = tf.random.normal([10, 100])
generated_image = generator(noise, training=False)
true_image = df.sample(10).to_numpy().reshape((10, df.shape[1]))
plt.imshow(generated_image, cmap='gray')
plt.show()
plt.imshow(true_image, cmap='gray')
plt.show()
print(discriminator(generated_image), discriminator(true_image))
