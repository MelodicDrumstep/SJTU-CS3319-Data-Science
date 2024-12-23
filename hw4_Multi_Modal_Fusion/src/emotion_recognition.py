import sys
import os
sys.path.append(os.path.abspath("util"))

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau 
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_subject_data
import logging

from feature_level_fusion_transformer import FeatureLevelFusionTransformer
from decision_level_fusion_transformer import DecisionLevelFusionTransformer
from bimodal_deep_autoencoder import BiModalDeepAutoencoder

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
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

def train_model_BDAE(model, train_loader, criterion_reconstruction, criterion_classification, optimizer, num_epochs=20):
    """
    Train the given model with both reconstruction and classification losses
    :param model: PyTorch model to train
    :param train_loader: DataLoader for the training data
    :param criterion_reconstruction: Loss function for reconstruction
    :param criterion_classification: Loss function for classification
    :param optimizer: Optimizer
    :param num_epochs: Number of training epochs
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for eeg_data, eye_data, labels in train_loader:
            optimizer.zero_grad()
            reconstructed1, reconstructed2, classification_output = model(eeg_data, eye_data)
            loss_reconstruction = criterion_reconstruction(reconstructed1, eeg_data) + criterion_reconstruction(reconstructed2, eye_data)
            loss_classification = criterion_classification(classification_output, labels)
            loss = loss_reconstruction + loss_classification
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the given PyTorch model and compute both accuracy and loss
    :param model: PyTorch model to evaluate
    :param test_loader: DataLoader for the test data
    :param criterion: Loss function used to calculate the loss
    :return: Accuracy and loss of the model on the test data
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for eeg_data, eye_data, labels in test_loader:
            outputs = model(eeg_data, eye_data)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()  # Add current loss to total loss
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)  # Compute average loss

    print(f"[evaluate_model] validation_loss is {average_loss}")
    return accuracy, average_loss

def evaluate_model_BDAE(model, test_loader, criterion_classification):
    """
    Evaluate the given PyTorch model and compute both accuracy and loss
    :param model: PyTorch model to evaluate
    :param test_loader: DataLoader for the test data
    :param criterion_classification: Loss function for classification
    :return: Accuracy and loss of the model on the test data
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for eeg_data, eye_data, labels in test_loader:
            outputs = model(eeg_data, eye_data)
            _, _, classification_output = outputs
            
            loss = criterion_classification(classification_output, labels)
            total_loss += loss.item()  # Add current loss to total loss
            
            _, predicted = torch.max(classification_output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)  # Compute average loss
    
    print(f"[evaluate_model_BDAE] validation_loss is {average_loss}")
    return accuracy, average_loss

def run_subject_validation(model_type, data_path='dataset', learning_rate=0.0001, 
                           hidden_size=64, latent_dim=32, output_size=3, 
                           num_heads=4, num_layers=2, lr_scheduler_type='StepLR',
                           num_epochs=20):
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

        if model_type == 'BiModalDeepAutoencoder':
            train_data_eeg = (train_data_eeg - train_data_eeg.mean()) / train_data_eeg.std()
            train_data_eye = (train_data_eye - train_data_eye.mean()) / train_data_eye.std()

        # Create DataLoader for training and testing
        train_loader = DataLoader(TensorDataset(train_data_eeg, train_data_eye, train_labels), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_data_eeg, test_data_eye, test_labels), batch_size=32, shuffle=False)

        # Initialize the model based on the specified model type
        if model_type == 'FeatureLevelFusionTransformer':
            model = FeatureLevelFusionTransformer(input_size1=train_data_eeg.shape[1], input_size2=train_data_eye.shape[1], 
                                                  hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, 
                                                  output_size=output_size)
        elif model_type == 'DecisionLevelFusion':
            model = DecisionLevelFusionTransformer(
                input_size_eeg=train_data_eeg.shape[1], 
                input_size_eye=train_data_eye.shape[1], 
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                output_size=output_size
            )

        elif model_type == 'BiModalDeepAutoencoder':
            model = BiModalDeepAutoencoder(
                input_size1=train_data_eeg.shape[1], 
                input_size2=train_data_eye.shape[1], 
                hidden_size=hidden_size, 
                latent_dim=latent_dim, 
                num_classes=output_size  # Ensure num_classes is provided
            )
        else:
            raise ValueError("Unsupported model type")
        
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        if lr_scheduler_type == 'StepLR':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Every 10 epochs, decrease the LR by 0.1
        elif lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # Reduce LR when loss plateaus
        else:
            raise ValueError("Unsupported learning rate scheduler")

        if model_type == 'BiModalDeepAutoencoder':
            # Define loss functions for reconstruction and classification
            criterion_reconstruction = nn.MSELoss()
            criterion_classification = nn.CrossEntropyLoss()
            # Train the model
            train_model_BDAE(model, train_loader, criterion_reconstruction, criterion_classification, optimizer, num_epochs=num_epochs)
            accuracy, validation_loss = evaluate_model_BDAE(model, test_loader, criterion_classification)
        else:
                # Define loss and optimizer
            criterion = nn.CrossEntropyLoss() 
            train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
            accuracy, validation_loss = evaluate_model(model, test_loader, criterion)

        scheduler.step(validation_loss)

        # Evaluate the model
        accuracies.append(accuracy)
        print(f"Accuracy for subject {subject_id}: {accuracy:.2f}")

    # Print average accuracy
    print(f"Average accuracy: {np.mean(accuracies):.2f}, std: {np.std(accuracies):.2f}")

    return float(np.mean(accuracies))

def grid_search(model_type, data_path='dataset', param_grid=None):
    """
    Perform a grid search over hyperparameter configurations.
    :param model_type: The type of model ('FeatureLevelFusionTransformer', 'DecisionLevelFusion', 'BiModalDeepAutoencoder')
    :param data_path: Path to the dataset
    :param param_grid: Dictionary with hyperparameter names as keys and lists of parameter values to try.
    :param num_epochs: Number of epochs to train each configuration
    """
    print(f"[grid_search] model_type is {model_type}")
    param_combinations = list(itertools.product(*param_grid.values()))
    best_accuracy = 0.0
    best_params = {}

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Evaluating configuration: {param_dict}")

        accuracy = run_subject_validation(
            model_type=model_type,
            data_path=data_path,
            learning_rate=param_dict['learning_rate'],
            hidden_size=param_dict['hidden_size'],
            latent_dim=param_dict['latent_dim'],
            output_size=param_dict['output_size'],
            num_heads=param_dict['num_heads'],
            num_layers=param_dict['num_layers'],
            lr_scheduler_type=param_dict['lr_scheduler_type'],
            num_epochs=param_dict['num_epochs']
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = param_dict

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    return best_params, best_accuracy

if __name__ == "__main__":
    # # Run subject-dependent validation using the specified model
    # run_subject_validation(model_type='FeatureLevelFusionTransformer')
    # # run_subject_validation(model_type='DecisionLevelFusion')
    # run_subject_validation(model_type='BiModalDeepAutoencoder')

    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_size': [64, 128, 256],
        'latent_dim': [32, 64],
        'output_size': [3],
        'num_heads': [4, 8],
        'num_layers': [2, 4],
        'lr_scheduler_type': ['StepLR', 'ReduceLROnPlateau'],
        "num_epochs" : [1, 5, 10, 20, 30]
    }

    # Perform grid search
    best_params, best_accuracy = grid_search(model_type='FeatureLevelFusionTransformer', param_grid=param_grid)
    print(f"Best Hyperparameters: {best_params}, Best Accuracy: {best_accuracy:.4f}")
