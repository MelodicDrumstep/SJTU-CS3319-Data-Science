import sys
import os
sys.path.append(os.path.abspath("util"))

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_seed_data_npy, reshape_to_spatial
from utils import channelID2str, channel_matrix

from mlp import MLPModel
from cnn import CNNModel
from lstm import LSTMModel
from rnn import RNNModel

def train_model(model, train_loader, criterion, optimizer, num_epochs=30):
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


def run_cross_subject_validation(model_type, data_path='dataset', learning_rate=0.0001, 
                                 MLP_hidden_sizes=[256, 128, 64], MLP_dropout_ratio=0.3,
                                 LSTM_hidden_size=64, LSTM_num_layers=2,
                                 RNN_hidden_size=64, RNN_num_layers=2):
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

        # Only reshape to spatial dimensions if using CNN model
        if model_type == 'CNN':
            train_tensor = torch.tensor(reshape_to_spatial(train_data, channel_matrix, channelID2str), dtype=torch.float32)
            test_tensor = torch.tensor(reshape_to_spatial(test_data, channel_matrix, channelID2str), dtype=torch.float32)

        # Create DataLoader for training and testing
        train_loader = DataLoader(TensorDataset(train_tensor, train_labels_tensor), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_tensor, test_labels_tensor), batch_size=32, shuffle=False)

        # Initialize the model
        if model_type == 'MLP':
            input_size = train_data.shape[1] * train_data.shape[2]  # Corrected to match flattened size
            #print(f"MLP input size: {input_size}") 
            model = MLPModel(input_size, hidden_sizes=MLP_hidden_sizes, dropout_ratio=MLP_dropout_ratio)
        elif model_type == 'CNN':
            input_channels = 1
            train_tensor = train_tensor.reshape(train_tensor.shape[0], 1, 62, 5)
            test_tensor = test_tensor.reshape(test_tensor.shape[0], 1, 62, 5)
            model = CNNModel(input_channels)
        elif model_type == 'LSTM':
            input_size = train_data.shape[2]
            model = LSTMModel(input_size, LSTM_hidden_size, LSTM_num_layers)
        elif model_type == 'RNN':
            input_size = train_tensor.shape[2]
            model = RNNModel(input_size, RNN_hidden_size, RNN_num_layers)
        else:
            raise ValueError("Unsupported model type")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_model(model, train_loader, criterion, optimizer)

        # Evaluate the model
        accuracy = evaluate_model(model, test_loader)
        accuracies.append(accuracy)
        print(f"Accuracy for subject {test_idx + 1}: {accuracy:.2f}")

    # Print average accuracy
    print(f"Cross-subject accuracy mean for {model_type}: {np.mean(accuracies):.2f}, std: {np.std(accuracies):.2f}")

def grid_search_MLP(hidden_size_index_low, hidden_size_index_high, dropout_index_low, dropout_index_high):
    for hidden_size_index in range(hidden_size_index_low, hidden_size_index_high):
        # 256, 128, 64
        hidden_size_1 = hidden_size_index * 4
        hidden_size_2 = hidden_size_index * 2
        hidden_size_3 = hidden_size_index
        for dropout_index in range(dropout_index_low, dropout_index_high):
            dropout_ratio = dropout_index * 1.0 / 10
            print(f"hidden_size_index : {hidden_size_index}")
            print(f"dropout_index : {dropout_index}\n")
            run_cross_subject_validation(model_type='MLP', 
                                            MLP_hidden_sizes=[hidden_size_1, hidden_size_2, hidden_size_3],
                                            MLP_dropout_ratio=dropout_ratio)
            
def grid_search_LSTM(hidden_size_index_low, hidden_size_index_high, num_layers_index_low, num_layers_index_high):
    for hidden_size_index in range(hidden_size_index_low, hidden_size_index_high):
        hidden_size = hidden_size_index * 8
        for num_layers in range(num_layers_index_low, num_layers_index_high):
            print(f"hidden_size_index : {hidden_size_index}")
            print(f"num_layers : {num_layers}\n")
            run_cross_subject_validation(model_type='LSTM', 
                                            LSTM_hidden_size=hidden_size,
                                            LSTM_num_layers=num_layers)

def grid_search_RNN(hidden_size_index_low, hidden_size_index_high, num_layers_index_low, num_layers_index_high):
    for hidden_size_index in range(hidden_size_index_low, hidden_size_index_high):
        hidden_size = hidden_size_index * 8
        for num_layers in range(num_layers_index_low, num_layers_index_high):
            print(f"hidden_size_index : {hidden_size_index}")
            print(f"num_layers : {num_layers}\n")
            run_cross_subject_validation(model_type='RNN', 
                                            RNN_hidden_size=hidden_size,
                                            RNN_num_layers=num_layers)

if __name__ == "__main__":
    # run_cross_subject_validation(model_type='MLP')
    # run_cross_subject_validation(model_type='RNN')
    # run_cross_subject_validation(model_type='LSTM')
    #run_cross_subject_validation(model_type='CNN')

    #grid_search_MLP(1, 100, 2, 8)
    #grid_search_LSTM(1, 20, 1, 8)
    grid_search_RNN(1, 6, 3, 7)
