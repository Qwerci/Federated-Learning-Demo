import json
import os
import numpy as np

log_dir = 'logs'

def load_logs(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    logs = [json.loads(line) for line in lines]
    return logs

def evaluate_logs():
    fit_logs = []
    evaluate_logs = []

    for file_name in os.listdir(log_dir):
        if 'fit' in file_name:
            fit_logs.extend(load_logs(os.path.join(log_dir, file_name)))
        elif 'evaluate' in file_name:
            evaluate_logs.extend(load_logs(os.path.join(log_dir, file_name)))

    # Calculate average metrics
    avg_fit_loss = np.mean([log['loss'] for log in fit_logs])
    avg_fit_accuracy = np.mean([log['accuracy'] for log in fit_logs])
    avg_evaluate_loss = np.mean([log['loss'] for log in evaluate_logs])
    avg_evaluate_accuracy = np.mean([log['accuracy'] for log in evaluate_logs])

    print(f"Average Fit Loss: {avg_fit_loss}")
    print(f"Average Fit Accuracy: {avg_fit_accuracy}")
    print(f"Average Evaluate Loss: {avg_evaluate_loss}")
    print(f"Average Evaluate Accuracy: {avg_evaluate_accuracy}")

    return avg_evaluate_loss, avg_evaluate_accuracy

if __name__ == "__main__":
    test_loss, test_accuracy = evaluate_logs()
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
