import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=3):
        """
        Create an LSTM model
        :param input_size: Number of input features
        :param hidden_size: Number of hidden units in each LSTM layer
        :param num_layers: Number of LSTM layers
        :param num_classes: Number of output classes
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        # Fully connected layer
        out = self.fc(out)
        out = self.softmax(out)
        return out
