import sys
import os
sys.path.append(os.path.abspath("util"))

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_seed_data_npy
import logging

from dann import DANNModel

logging.basicConfig(level=logging.DEBUG)

def train_DA(model, train_loader, test_loader_for_train, criterion, optimizer, num_epochs=20, alpha=0.1, use_domain_classifier=True, mode="DA"):
    """
    Train the given model
    :param model: PyTorch model to train
    :param train_loader: DataLoader for the training data
    :param test_loader: DataLoader for the test data (only for DA)
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param num_epochs: Number of training epochs
    :param alpha: Gradient reversal scaling factor
    :param use_domain_classifier: Whether to use the domain classifier (default True)
    :param mode: Mode of training ('DA' or 'DG')
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels, domain_labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten input
            optimizer.zero_grad()

            # Forward pass
            if use_domain_classifier:
                label_pred, domain_pred = model(inputs, alpha)

                # Ensure domain_pred has shape (batch_size, 2) for binary classification
                domain_loss = criterion(domain_pred, domain_labels)  # Domain loss for training
                label_loss = criterion(label_pred, labels)
                total_loss = label_loss + alpha * domain_loss
            else:
                label_pred = model(inputs, alpha)
                total_loss = criterion(label_pred, labels)

            # Backward pass
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        if epoch % 5 == 0 and use_domain_classifier:
            for inputs, domain_labels in test_loader_for_train:
                inputs = inputs.view(inputs.size(0), -1)  # Flatten input
                optimizer.zero_grad()

                _, domain_pred = model(inputs, alpha)
                # Ensure domain_pred has shape (batch_size, 2) for binary classification
                domain_loss = criterion(domain_pred, domain_labels)  # Domain loss for training
                total_loss = alpha * domain_loss

                # Backward pass
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

        # Print loss for each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


def train_DG(model, train_loader, criterion, optimizer, alpha, num_epochs=20, use_domain_classifier=True):
    """
    Train the model for domain generalization
    :param model: PyTorch model to train
    :param train_loaders: List of DataLoaders for training data from multiple domains
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param num_epochs: Number of training epochs
    :param use_domain_classifier: Whether to use the domain classifier (default True)
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, domain_labels in train_loader:
            optimizer.zero_grad()

            if use_domain_classifier:
                label_pred, domain_pred = model(inputs, alpha)
                # Domain loss (cross-entropy)
                # logging.debug(f"[train_DG] domain_pred is {domain_pred}, domain_labels is {domain_labels}")
                domain_loss = criterion(domain_pred, domain_labels)
                loss = criterion(label_pred, labels) + alpha * domain_loss
            else:
                label_pred = model(inputs, alpha)
                loss = criterion(label_pred, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / sum(len(loader) for loader in train_loader)}")


def evaluate_model(model, test_loader, use_domain_classifier=True):
    """
    Evaluate the given PyTorch model
    :param model: PyTorch model to evaluate
    :param test_loader: DataLoader for the test data
    :param use_domain_classifier: Whether to use the domain classifier (default True)
    :return: Accuracy of the model on the test data
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1)  # (batch_size, 62, 5) flatten to (batch_size, 310)
            if use_domain_classifier:
                outputs = model(inputs)
                label_pred = outputs[0]  # extract label_pred
            else:
                label_pred = model(inputs)
            _, predicted = torch.max(label_pred, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def run_cross_subject_validation(model_type, mode, data_path='dataset', learning_rate=0.0001, alpha=0.1,
                                 feature_hidden_dims=[128, 64], label_output_dim=3, domain_hidden_dims=[32],
                                 use_domain_classifier=True):
    """
    Run cross-subject leave-one-out validation using the given model type
    :param model_type: Type of model being used (e.g., 'DANN')
    :param mode: Mode of training ('DA' or 'DG')
    :param data_path: Path to the dataset
    :param learning_rate: Learning rate for the optimizer
    :param alpha: Gradient reversal scaling factor for DANN
    :param feature_hidden_dims: Hidden dimensions for the feature extractor
    :param label_output_dim: Output dimension for the label classifier
    :param domain_hidden_dims: Hidden dimensions for the domain classifier
    :param domain_output_dim: Output dimension for the domain classifier
    :param use_domain_classifier: Whether to use the domain classifier (default True)
    """
    subject_data = load_seed_data_npy(data_path)
    num_subjects = len(subject_data)

    accuracies = []
    for test_idx in range(num_subjects):
        # Prepare training and testing data
        test_data, test_labels = subject_data[test_idx]
        train_data = []
        train_labels = []
        train_domain_labels = []

        # Gather all other subjects as training data
        for i in range(num_subjects):
            if i != test_idx:
                logging.debug(f"subject_data[{i}][0].shape is {subject_data[i][0].shape}")
                logging.debug(f"subject_data[{i}][1].shape is {subject_data[i][1].shape}")
                train_data.append(subject_data[i][0])  # Append data
                train_labels.append(subject_data[i][1])  # Append labels
                if mode == 'DG':
                    source_domain_label = i if i < test_idx else i - 1
                elif mode == "DA":
                    source_domain_label = 0
                else:
                    raise ValueError(mode)
                logging.debug(f"source_domain_label is {source_domain_label}")
                domain_label_np = np.full(subject_data[i][0].shape[0], source_domain_label, dtype=np.int32)
                train_domain_labels.append(domain_label_np)
                logging.debug(f"(domain_label.shape is {domain_label_np.shape}")


        # Concatenate training data
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_domain_labels = np.concatenate(train_domain_labels, axis=0)

        # Convert data to PyTorch tensors
        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        train_domain_labels_tensor = torch.tensor(train_domain_labels, dtype=torch.long)
        test_tensor = torch.tensor(test_data, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

        train_tensor = train_tensor.view(train_tensor.size(0), -1)  # (62, 5) flatten to 310
        test_tensor = test_tensor.view(test_tensor.size(0), -1)
        if mode == 'DA':
            test_domain_labels = np.full(test_data.shape[0], 1, dtype=np.int32)
            test_domain_tensor = torch.tensor(test_domain_labels, dtype=torch.long)

        logging.debug(f"train_tensor.shape is {train_tensor.shape}")
        logging.debug(f"train_labels_tensor.shape is {train_labels_tensor.shape}")
        logging.debug(f"train_domain_labels_tensor.shape is {train_domain_labels_tensor.shape}")

        # Create DataLoader for training and testing, add domain_labels to DataLoader for DG
        if mode == 'DG':
            logging.debug(f"mode is DG, packing the dataset")
            train_loader = DataLoader(TensorDataset(train_tensor, train_labels_tensor, train_domain_labels_tensor), batch_size=32, shuffle=True)
            test_loader = DataLoader(TensorDataset(test_tensor, test_labels_tensor), batch_size=32, shuffle=False)
        else:
            train_loader = DataLoader(TensorDataset(train_tensor, train_labels_tensor, train_domain_labels_tensor), batch_size=32, shuffle=False)
            test_loader = DataLoader(TensorDataset(test_tensor, test_labels_tensor), batch_size=32, shuffle=False)
            test_loader_for_train = DataLoader(TensorDataset(test_tensor, test_domain_tensor), batch_size=32, shuffle=False)

        # Initialize the model
        if model_type == 'DANN':
            model = DANNModel(
                input_dim=310,
                feature_hidden_dims=feature_hidden_dims,
                label_output_dim=label_output_dim,
                domain_hidden_dims=domain_hidden_dims,
                mode=mode,
                use_domain_classifier=use_domain_classifier  # Pass the use_domain_classifier flag here
            )
        else:
            raise ValueError("Unsupported model type")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        if mode == "DG":
            train_DG(model, train_loader, criterion, optimizer, alpha=alpha, use_domain_classifier=use_domain_classifier)
        elif mode == "DA":
            train_DA(model, train_loader, test_loader_for_train, criterion, optimizer, alpha=alpha, use_domain_classifier=use_domain_classifier)
        else:
            raise ValueError(mode)      

        # Evaluate the model
        accuracy = evaluate_model(model, test_loader, use_domain_classifier=use_domain_classifier)
        accuracies.append(accuracy)
        print(f"Accuracy for subject {test_idx + 1}: {accuracy:.2f}")

    # Print average accuracy
    print(f"Cross-subject accuracy mean for {model_type}: {np.mean(accuracies):.2f}, std: {np.std(accuracies):.2f}")
    return np.mean(accuracies)


def grid_search_DANN(mode, feature_hidden_dims_options, domain_hidden_dims_options, use_domain_classifier=True):
    """
    Grid search for DANN model with feature_hidden_dims, domain_hidden_dims, and dropout_ratio.
    :param mode: Mode of training ('DA' or 'DG')
    :param feature_hidden_dims_options: List of options for feature_hidden_dims
    :param domain_hidden_dims_options: List of options for domain_hidden_dims
    :param use_domain_classifier: Whether to use the domain classifier (default True)
    """
    best_accuracy = 0.0
    best_params = {}

    for feature_hidden_dims in feature_hidden_dims_options:
        for domain_hidden_dims in domain_hidden_dims_options:
            print(f"mode is {mode}, use_domain_classifier={use_domain_classifier}")
            print(f"Testing feature_hidden_dims: {feature_hidden_dims}, domain_hidden_dims: {domain_hidden_dims}")

            # Run cross-subject validation with current hyperparameters
            accuracy = run_cross_subject_validation(
                model_type='DANN',
                mode=mode,
                feature_hidden_dims=feature_hidden_dims,
                domain_hidden_dims=domain_hidden_dims,
                learning_rate=0.0001,
                alpha=0.01,
                use_domain_classifier=use_domain_classifier
            )

            # Track the best parameters
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'feature_hidden_dims': feature_hidden_dims,
                    'domain_hidden_dims': domain_hidden_dims
                }

    print(f"Best accuracy: {best_accuracy:.2f} with parameters: {best_params}")


if __name__ == "__main__":
    grid_search_DANN("DA", [[256, 128]], [[64]])

    grid_search_DANN("DG", [[128, 64]], [[64]])
    grid_search_DANN("DA", [[256, 128]], [[64]], False)
    grid_search_DANN("DG", [[128, 64]], [[64]], False)
