import tensorflow as tf
from keras import layers

def create_model():
    return tf.keras.models.Sequential([
        layers.Input(shape=(28,28, 1)),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation ='relu'),
        layers.Dense(10, activation='softmax')
    ])

