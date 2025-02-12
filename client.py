import threading
from model import create_model
from dataL import client_data, x_test, y_test, num_client
from flw import FlowerClient
from flwr.client import start_numpy_client

def run_client(client_id):
    model = create_model()
    x_train, y_train = client_data[f'client_{client_id}']
    x_val, y_val = x_test[:1000], y_test[:1000]
    client = FlowerClient(model, x_train, y_train, x_val, y_val, client_id)
    start_numpy_client(server_address="localhost:8080", client=client)

# Start clients in separate threads
threads = []
for i in range(num_client):
    t = threading.Thread(target=run_client, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
