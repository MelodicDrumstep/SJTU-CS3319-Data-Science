import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes=3, hidden_sizes=[128, 64]):
        """
        Create an MLP model
        :param input_size: Number of input features
        :param hidden_sizes: List of hidden layer sizes
        :param num_classes: Number of output classes
        """
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x