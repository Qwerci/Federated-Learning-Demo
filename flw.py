import json
import os
from flwr.client import NumPyClient

class FlowerClient(NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val, client_id):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.client_id = client_id
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)

    def get_perameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        log_path = os.path.join(self.log_dir, f'client_{self.client_id}_fit.json')
        with open(log_path, 'a') as f:
            json.dump(history.history, f)
            f.write('\n')
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        log_path = os.path.join(self.log_dir, f'client_{self.client_id}_evaluate.json')
        with open(log_path, 'a') as f:
            json.dump({'loss': loss, 'accuracy': accuracy}, f)
            f.write('\n')
        return loss, len(self.x_val), {"accuracy": accuracy}
