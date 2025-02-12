from model import create_model
from dataL import num_client
from flwr.server.strategy import FedAvg
from flwr.server import start_server
from flwr.common import ndarrays_to_parameters
from flwr.server.app import ServerConfig

def get_model_fn():
    def model_fn():
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    return model_fn

strategy = FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=num_client,
    evaluate_fn=None,
    on_fit_config_fn=None,
    initial_parameters=ndarrays_to_parameters(create_model().get_weights())
)

# Create ServerConfig object
config = ServerConfig(num_rounds=5)

# Start server
start_server(server_address="localhost:8080", config=config, strategy=strategy)
