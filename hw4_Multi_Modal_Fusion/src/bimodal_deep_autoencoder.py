import torch
import torch.nn as nn

class BiModalDeepAutoencoder(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, latent_dim):
        """
        BiModal Deep Autoencoder (BDAE)
        :param input_size1: EEG feature size (input size for modality 1)
        :param input_size2: EOG feature size (input size for modality 2)
        :param hidden_size: Hidden size of the encoder layers
        :param latent_dim: Latent dimension (the compressed representation)
        """
        super(BiModalDeepAutoencoder, self).__init__()
        
        # Encoder for modality 1 (EEG)
        self.encoder1 = nn.Sequential(
            nn.Linear(input_size1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim)
        )
        
        # Encoder for modality 2 (EOG)
        self.encoder2 = nn.Sequential(
            nn.Linear(input_size2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim)
        )
        
        # Decoder (to reconstruct the inputs from the combined latent representation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size1 + input_size2)  # Reconstruct both EEG and EOG inputs
        )
    
    def forward(self, x1, x2):
        """
        Forward pass through the BiModal Deep Autoencoder
        :param x1: EEG data (batch_size, input_size1)
        :param x2: EOG data (batch_size, input_size2)
        :return: Reconstructed data (batch_size, input_size1 + input_size2)
        """
        # Encode both modalities
        latent1 = self.encoder1(x1)
        latent2 = self.encoder2(x2)
        
        # Concatenate the latent representations of both modalities
        concatenated_latent = torch.cat((latent1, latent2), dim=1)  # Concatenate along the feature dimension
        
        # Decode the concatenated latent representation to reconstruct both EEG and EOG inputs
        reconstructed = self.decoder(concatenated_latent)
        
        return reconstructed
