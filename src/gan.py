import numpy as np
import pandas as pd


from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras.metrics import Precision, Recall, AUC

from sklearn.model_selection import train_test_split
import tensorflow as tf

def generator_loss_log_d(y_true, y_pred):
    return -K.mean(K.log(y_pred + K.epsilon()))


class GAN:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim 
        self.generator = None
        self.discriminator = None
        self.gan = None

    # Define the generator network
    def _build_generator(self, latent_dim, output_dim):
        model = Sequential()
        model.add(Dense(64, input_shape=(latent_dim,)))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(output_dim, activation='sigmoid'))
        return model


    # Define the discriminator network
    def _build_discriminator(self, input_dim, learning_rate=0.0002, beta_1=0.5):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=beta_1), loss='binary_crossentropy',  metrics=[Precision(), Recall()])
        return model


    # GAN model combining generator and discriminator
    def _build_gan(self, learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999):
        self.discriminator.trainable = False
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2 = beta_2), loss=generator_loss_log_d)
        self.gan = model

    def _build_all(self, fraud_data):
        self.generator = self._build_generator(self.latent_dim, fraud_data.shape[1])
        self.discriminator = self._build_discriminator(fraud_data.shape[1])
        self.gan = self._build_gan(self.generator, self.discriminator)

    def _generate_fraud_samples(self, fraud_data, batch_size ):
        num_real_fraud = len(fraud_data)
        self.discriminator.trainable = True
        self.generator.trainable = False
        # Select random real fraud samples
        real_fraud_samples = fraud_data[np.random.randint(0, num_real_fraud, batch_size)]
        # Generate fake fraud samples using the generator
        noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
        fake_fraud_samples = self.generator.predict(noise)
        return real_fraud_samples, fake_fraud_samples
    
    def _train_discriminator(self, batch_size):
        # Create labels for real and fake fraud samples
        real_fraud_samples, fake_fraud_samples = self._generate_fraud_samples() #TODO: maek from this the function
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Convert real_fraud_samples and fake_fraud_samples to the correct data type if necessary
        real_fraud_samples = real_fraud_samples.astype(np.float32)  # or np.float64
        fake_fraud_samples = fake_fraud_samples.astype(np.float32)  # or np.float64

        # Train the discriminator on real and fake fraud samples
        d_loss_real = self.discriminator.train_on_batch(real_fraud_samples, real_labels)
        d_loss_fake = self.discriminator.train_on_batch(fake_fraud_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss
    
    def _train_generator(self, batch_size):
        # Train generator (freeze discriminator)
        self.discriminator.trainable = False
        self.generator.trainable = True
        # Generate fake fraud samples and create labels for training the generator
        noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
        valid_labels = np.ones((batch_size, 1))
        # Train the generator to generate samples that "fool" the discriminator
        g_loss = self.gan.train_on_batch(noise, valid_labels)
        return g_loss
    
    def _generate_synthetic_data(self, fraud_data, num_synthetic_samples, epochs = 1000, batch_size = 32):
        self._make_all(fraud_data)
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            # Training loop for the GAN
            for epoch in range(epochs):
                # Train discriminator (freeze generator)
                d_loss = self._train_discriminator(batch_size)
                g_loss = self._train_generator(batch_size)
                # Print the progress
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch} - D Loss: {d_loss} - G Loss: {g_loss}")
            # After training, use the generator to create synthetic fraud data
            noise = np.random.normal(0, 1, size=(num_synthetic_samples, self.latent_dim))
            synthetic_fraud_data = self.generator.predict(noise)
            return synthetic_fraud_data