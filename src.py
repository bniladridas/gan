import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess the MNIST dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)

# Define the generator model
generator = keras.Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(7 * 7 * 128),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", activation="relu"),
    layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", activation="relu"),
    layers.Conv2DTranspose(1, (7, 7), padding="same", activation="sigmoid"),
])

# Define the discriminator model
discriminator = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
    layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu"),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid"),
])

# Compile the discriminator (use binary crossentropy loss for binary classification)
discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

# Combine the generator and discriminator to form the GAN
discriminator.trainable = False  # Freeze the discriminator during GAN training
gan_input = keras.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = keras.Model(gan_input, gan_output)
gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

# Training the GAN
epochs = 10000
batch_size = 64

for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=(batch_size, 100))
    generated_images = generator.predict(noise)

    real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, size=(batch_size, 100))
    labels_gan = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(noise, labels_gan)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

        # Save generated images at certain intervals
        if epoch % 1000 == 0:
            generated_images = generated_images * 255.0
            generated_images = generated_images.astype('uint8')

            for i in range(batch_size):
                plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                plt.axis('off')
                plt.savefig(f"gan_generated_image_epoch_{epoch}_sample_{i}.png")
                plt.close()

