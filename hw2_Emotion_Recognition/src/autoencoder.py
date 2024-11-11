import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Create an Autoencoder model
        :param input_size: Number of input features
        :param hidden_size: Number of units in the hidden layer
        """
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)
        # Decode the encoded representation
        decoded = self.decoder(encoded)
        return decoded
