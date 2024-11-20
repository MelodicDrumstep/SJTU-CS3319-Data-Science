import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes=3):
        """
        Create a CNN model
        :param input_channels: Number of input channels
        :param num_classes: Number of output classes
        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Calculate the size of the input for fc1 dynamically
        # After two conv layers and two pool layers, output size is (batch_size, 64, 15, 1)
        self.fc1 = nn.Linear(64 * 15 * 1, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # Output shape: (batch_size, 32, 62, 5)
        x = self.relu1(x)
        x = self.pool1(x)  # Output shape: (batch_size, 32, 31, 2)
        x = self.conv2(x)  # Output shape: (batch_size, 64, 31, 2)
        x = self.relu2(x)
        x = self.pool2(x)  # Output shape: (batch_size, 64, 15, 1)
        x = self.flatten(x)  # Output shape: (batch_size, 64 * 15 * 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
