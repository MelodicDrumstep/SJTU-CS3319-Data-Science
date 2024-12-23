import torch
import torch.nn as nn

class BiModalDeepAutoencoder(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, latent_dim, num_classes):
        """
        BiModal Deep Autoencoder (BDAE) with Classification
        :param input_size1: EEG feature size (input size for modality 1)
        :param input_size2: EOG feature size (input size for modality 2)
        :param hidden_size: Hidden size of the encoder layers
        :param latent_dim: Latent dimension (the shared representation size)
        :param num_classes: Number of classes for classification
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
        
        # Decoder for modality 1 (EEG reconstruction)
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size1)
        )
        
        # Decoder for modality 2 (EOG reconstruction)
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size2)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x1, x2):
        """
        Forward pass through the BiModal Deep Autoencoder with classification
        :param x1: EEG data (batch_size, input_size1)
        :param x2: EOG data (batch_size, input_size2)
        :return: Reconstructed data for both modalities and classification output
        """
        # Encode both modalities
        latent1 = self.encoder1(x1)  # Latent representation for modality 1
        latent2 = self.encoder2(x2)  # Latent representation for modality 2
        
        # Shared representation (average of the two latent representations)
        shared_latent = (latent1 + latent2) / 2
        
        # Decode the shared representation back to the original input space for each modality
        reconstructed1 = self.decoder1(shared_latent)  # Reconstruct EEG data
        reconstructed2 = self.decoder2(shared_latent)  # Reconstruct EOG data
        
        # Classification output
        classification_output = self.classifier(shared_latent)
        
        return reconstructed1, reconstructed2, classification_output
