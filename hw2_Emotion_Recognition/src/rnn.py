import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=3):
        """
        Create an RNN model
        :param input_size: Number of input features
        :param hidden_size: Number of hidden units in each RNN layer
        :param num_layers: Number of RNN layers
        :param num_classes: Number of output classes
        """
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # RNN layer
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        # Check the dimensions of the output
        if out.dim() == 3:
            # Take the output of the last time step
            out = out[:, -1, :]  # (batch_size, hidden_size)
        # Fully connected layer
        out = self.fc(out)  # Final output: (batch_size, num_classes)
        return out
