import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes=3, hidden_sizes=[256, 128, 64], dropout_ratio=0.3):
        """
        Create an MLP model
        :param input_size: Number of input features
        :param hidden_sizes: List of hidden layer sizes
        :param num_classes: Number of output classes
        """
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_ratio) 
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_ratio) 
        
        # Third hidden layer
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_ratio)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)

    def forward(self, x):
        x = self.flatten(x)
        
        # First layer: FC -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second layer: FC -> ReLU -> Dropout
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Third layer: FC -> ReLU -> Dropout
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output layer (no activation function here, we apply Softmax outside if needed)
        x = self.fc4(x)
        return x
