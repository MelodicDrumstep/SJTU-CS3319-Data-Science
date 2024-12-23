import sys
import os
sys.path.append(os.path.abspath("util"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_subject_data

from feature_level_fusion_transformer import FeatureLevelFusionTransformer
from decision_level_fusion_transformer import DecisionLevelFusionTransformer
from bimodal_deep_autoencoder import BiModalDeepAutoencoder

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
        for eeg_data, eye_data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(eeg_data, eye_data)
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
        for eeg_data, eye_data, labels in test_loader:
            outputs = model(eeg_data, eye_data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def run_subject_validation(model_type, data_path='dataset', learning_rate=0.0001, 
                           hidden_size=64, latent_dim=32, output_size=3, 
                           num_heads=4, num_layers=2):
    """
    Run leave-one-out validation for each subject
    :param model_type: Type of model being used (e.g., 'FeatureLevelFusionTransformer', 'DecisionLevelFusion', 'BiModalDeepAutoencoder')
    :param data_path: Path to the dataset
    """
    subject_data = load_subject_data(data_path)
    
    accuracies = []
    for subject_id, data in subject_data.items():
        # Prepare training and testing data for the current subject
        test_data_eeg = torch.tensor(data['test_data_eeg'], dtype=torch.float32)
        test_data_eye = torch.tensor(data['test_data_eye'], dtype=torch.float32)
        test_labels = torch.tensor(data['test_labels'], dtype=torch.long)
        
        train_data_eeg = []
        train_data_eye = []
        train_labels = []
        
        # Use data from all other subjects as the training data
        for other_subject_id, other_data in subject_data.items():
            if other_subject_id != subject_id:
                train_data_eeg.append(other_data['train_data_eeg'])
                train_data_eye.append(other_data['train_data_eye'])
                train_labels.append(other_data['train_labels'])
        
        # Concatenate training data
        train_data_eeg = torch.tensor(np.concatenate(train_data_eeg, axis=0), dtype=torch.float32)
        train_data_eye = torch.tensor(np.concatenate(train_data_eye, axis=0), dtype=torch.float32)
        train_labels = torch.tensor(np.concatenate(train_labels, axis=0), dtype=torch.long)

        # Create DataLoader for training and testing
        train_loader = DataLoader(TensorDataset(train_data_eeg, train_data_eye, train_labels), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_data_eeg, test_data_eye, test_labels), batch_size=32, shuffle=False)

        # Initialize the model based on the specified model type
        if model_type == 'FeatureLevelFusionTransformer':
            model = FeatureLevelFusionTransformer(input_size1=train_data_eeg.shape[1], input_size2=train_data_eye.shape[1], 
                                                  hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, 
                                                  output_size=output_size)
        elif model_type == 'DecisionLevelFusion':
            model = DecisionLevelFusionTransformer(input_size=train_data_eeg.shape[1] + train_data_eye.shape[1], hidden_size=hidden_size, 
                                         output_size=output_size)
        elif model_type == 'BiModalDeepAutoencoder':
            model = BiModalDeepAutoencoder(input_size1=train_data_eeg.shape[1], input_size2=train_data_eye.shape[1], 
                                           hidden_size=hidden_size, latent_dim=latent_dim)
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
        print(f"Accuracy for subject {subject_id}: {accuracy:.2f}")

    # Print average accuracy
    print(f"Average accuracy: {np.mean(accuracies):.2f}, std: {np.std(accuracies):.2f}")

if __name__ == "__main__":
    # Run subject-dependent validation using the specified model
    # run_subject_validation(model_type='FeatureLevelFusionTransformer')
    run_subject_validation(model_type='DecisionLevelFusion')
    run_subject_validation(model_type='BiModalDeepAutoencoder')
