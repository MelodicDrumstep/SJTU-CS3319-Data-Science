import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_seed_data_npy

from mlp import MLPModel
from cnn import CNNModel
from lstm import LSTMModel
from rnn import RNNModel
from autoencoder import Autoencoder


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the given model
    :param model: PyTorch model to train
    :param train_loader: DataLoader for the training data
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param num_epochs: Number of training epochs
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Only flatten the input if the model is MLP
            if isinstance(model, MLPModel):
                inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            #print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def evaluate_model(model, test_loader):
    """
    Evaluate the given PyTorch model
    :param model: PyTorch model to evaluate
    :param test_loader: DataLoader for the test data
    :return: Accuracy of the model on the test data
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def run_cross_subject_validation(model_type, data_path='dataset'):
    """
    Run cross-subject leave-one-out validation using the given model type
    :param model_type: Type of model being used (e.g., 'MLP', 'CNN', etc.)
    :param data_path: Path to the dataset
    """
    subject_data = load_seed_data_npy(data_path)
    num_subjects = len(subject_data)

    accuracies = []
    for test_idx in range(num_subjects):
        # Prepare training and testing data
        test_data, test_labels = subject_data[test_idx]
        train_data = []
        train_labels = []

        # Gather all other subjects as training data
        for i in range(num_subjects):
            if i != test_idx:
                train_data.append(subject_data[i][0])  # Append data
                train_labels.append(subject_data[i][1])  # Append labels

        # Concatenate training data
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # Convert data to PyTorch tensors
        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        test_tensor = torch.tensor(test_data, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

        # Create DataLoader for training and testing
        train_loader = DataLoader(TensorDataset(train_tensor, train_labels_tensor), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_tensor, test_labels_tensor), batch_size=32, shuffle=False)

        # Initialize the model
        if model_type == 'MLP':
            input_size = train_data.shape[1] * train_data.shape[2]  # Corrected to match flattened size
            #print(f"MLP input size: {input_size}") 
            model = MLPModel(input_size)
        elif model_type == 'CNN':
            input_channels = 1
            train_data = train_data.reshape(train_data.shape[0], 1, 62, 5)
            test_data = test_data.reshape(test_data.shape[0], 1, 62, 5)
            model = CNNModel(input_channels)
        elif model_type == 'LSTM':
            input_size = train_data.shape[1] * train_data.shape[2]
            hidden_size = 64
            num_layers = 2
            model = LSTMModel(input_size, hidden_size, num_layers)
        elif model_type == 'RNN':
            input_size = train_tensor.shape[2]
            hidden_size = 64
            num_layers = 2
            model = RNNModel(input_size, hidden_size, num_layers)
        elif model_type == 'Autoencoder':
            input_size = train_data.shape[1]
            hidden_size = 128
            model = Autoencoder(input_size, hidden_size)
        else:
            raise ValueError("Unsupported model type")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss() if model_type != 'Autoencoder' else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        train_model(model, train_loader, criterion, optimizer)

        # Evaluate the model
        accuracy = evaluate_model(model, test_loader)
        accuracies.append(accuracy)
        print(f"Accuracy for subject {test_idx + 1}: {accuracy:.2f}")

    # Print average accuracy
    print(f"Cross-subject accuracy mean: {np.mean(accuracies):.2f}, std: {np.std(accuracies):.2f}")


# Example usage
if __name__ == "__main__":
    #run_cross_subject_validation(model_type='MLP')
    run_cross_subject_validation(model_type='RNN')
