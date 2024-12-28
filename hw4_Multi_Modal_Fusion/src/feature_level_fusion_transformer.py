import torch
import torch.nn as nn

class FeatureLevelFusionTransformer(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, num_heads, num_layers, output_size, dropout=0.1):
        """
        Feature-level Fusion Transformer with Dropout
        :param input_size1: Size of the first modality (EEG features)
        :param input_size2: Size of the second modality (EOG features)
        :param hidden_size: Size of the hidden layers in the Transformer
        :param num_heads: Number of attention heads in the Transformer
        :param num_layers: Number of Transformer layers
        :param output_size: The number of output classes, 3 classes for emotions
        :param dropout: Dropout rate for regularization
        """
        super(FeatureLevelFusionTransformer, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Sequential(
            nn.Linear(input_size1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout) 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_size2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout) 
        )
        self.fc_transformer_input = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout) 
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Dropout(dropout)  
        )
    
    def forward(self, x1, x2):
        """
        Forward pass through the feature-level fusion transformer
        :param x1: EEG data (batch_size, seq_len, input_size1)
        :param x2: EOG data (batch_size, seq_len, input_size2)
        :return: Output of the model (batch_size, output_size)
        """
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = torch.cat((x1, x2), dim=-1)  
        x = self.fc_transformer_input(x) 
        # x = x.unsqueeze(1)
        x = self.transformer_encoder(x) 
        # x = x.squeeze(1) 
        x = self.fc_out(x)  
        
        return x
