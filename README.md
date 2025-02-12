# Federated Learning Demo with Flower

This project demonstrates a federated learning setup using the Flower framework with the MNIST dataset. Federated learning allows multiple clients to collaboratively train a machine learning model while keeping the training data decentralized. This is particularly useful for privacy-preserving applications.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Federated learning is a machine learning setting where multiple clients (e.g., mobile devices, hospitals) collaboratively train a model while keeping the training data decentralized. This project uses the Flower framework to implement federated learning with the MNIST dataset. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), split into 60,000 training images and 10,000 testing images.

## Features

- Federated learning setup using the Flower framework.
- Support for multiple clients to participate in the training process.
- Evaluation of the model on a test dataset.
- Visualization of training metrics and evaluation results using Streamlit.

## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/federated-learning-demo.git
   cd federated-learning-demo

2. **Create a Virtual Environment**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt

4. **Download the MNIST Dataset**:
    The MNIST dataset will be automatically downloaded and prepared when you run the server and client scripts.

## Usage
### Start the Server
1. Open a terminal and navigate to the project directory.
2. Run the server script to start the Flower server.
    ```sh
    python server.py

    The server will start and listen for client connections on localhost:8080.

### Start the Clients
1. Open a new terminal and navigate to the project directory.
2. Run the client script to start multiple clients in separate threads.
    ```sh
    python client.py

    This script will start multiple clients, each connecting to the server at localhost:8080. Wait for all clients to finish their training rounds.

### Evaluate the Results
1. After the federated learning process is complete, you can evaluate the logged results.
2. Open a new terminal and navigate to the project directory.
3. Run the evaluation script.
    ```sh
    python eval.py

    This script will read the logged results from the clients and calculate the average metrics.

### Visualize the Results
1. After evaluating the results, you can visualize them using the Streamlit app.
2. Open a new terminal and navigate to the project directory.
3. Run the Streamlit app.
    ```sh
    streamlit run stlit.py

    This will start a local web server, and you can view the results in your web browser by navigating to the provided URL (usually http://localhost:8501).

## Project Structure

- model.py: Defines the machine learning model architecture.
- dataL.py: Loads and prepares the MNIST dataset, and splits it into multiple clients.
- flw.py: Defines the Flower client class with methods for federated learning.
- server.py: Starts the Flower server and defines the federated learning strategy.
- client.py: Starts multiple clients to participate in the federated learning process.
- eval.py: Evaluates the logged results from the clients and calculates average metrics.
- stlit.py: Streamlit app to visualize training metrics and evaluation results.
- requirements.txt: List of Python dependencies for the project.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your fork.
4. Submit a pull request with a detailed description of your changes.

License
This project is licensed under the Apache License. See the LICENSE file for details.

