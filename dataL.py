import numpy as np
import tensorflow as tf
from keras import datasets
from sklearn.model_selection import train_test_split

# Load dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/ 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/ 255.0

# Shuffle data
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]


# Split into client 
num_client = 5
client_data = {}
for i in range(num_client):
    start_idx = int(len(x_train) * i / num_client)
    end_idx = int(len(x_train) * (i + 1) / num_client)
    client_x = x_train[start_idx:end_idx]
    client_y = y_train[start_idx:end_idx]
    client_data[f'client_{i}'] = (client_x, client_y)