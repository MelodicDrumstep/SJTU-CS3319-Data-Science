import torch
import torch.nn as nn

import logging

class DecisionLevelFusionTransformer(nn.Module):
    def __init__(self, input_size_eeg, input_size_eye, hidden_size, num_heads, num_layers, output_size):
        """
        Decision-Level Fusion with two independent Transformers
        :param input_size_eeg: The input size of EEG data
        :param input_size_eye: The input size of Eye tracking data
        :param hidden_size: The size of the hidden layers in the Transformer
        :param num_heads: The number of attention heads in the Transformer
        :param num_layers: The number of Transformer layers
        :param output_size: The number of output classes (e.g., 3 classes for emotions)
        """
        super(DecisionLevelFusionTransformer, self).__init__()

        # Linear layers to map EEG and Eye data to the same feature size
        self.eeg_projection = nn.Linear(input_size_eeg, hidden_size)
        self.eye_projection = nn.Linear(input_size_eye, hidden_size)

        # EEG Transformer
        self.eeg_transformer = nn.Transformer(
            d_model=hidden_size, 
            num_encoder_layers=num_layers,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True  
        )
        
        # Eye Transformer
        self.eye_transformer = nn.Transformer(
            d_model=hidden_size,  
            num_encoder_layers=num_layers,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True 
        )

        self.fc = nn.Linear(hidden_size * 2, output_size)  # Hidden size * 2 after concatenation

    def forward(self, eeg_data, eye_data):
        """
        Forward pass through the Decision-Level Fusion Transformer
        :param eeg_data: EEG data (batch_size, seq_len, input_size_eeg)
        :param eye_data: Eye tracking data (batch_size, seq_len, input_size_eye)
        :return: The fused output (final prediction)
        """
        # Project EEG and Eye data to the same feature size
        eeg_data = self.eeg_projection(eeg_data)  
        eye_data = self.eye_projection(eye_data)  
        eeg_output = self.eeg_transformer(eeg_data, eeg_data)  
        eye_output = self.eye_transformer(eye_data, eye_data)  

        # logging.debug(f"eeg_output shape: {eeg_output.shape}")
        # logging.debug(f"eye_output shape: {eye_output.shape}")

        if eeg_output.dim() == 3: 
            eeg_output = eeg_output[:, -1, :]  
        if eye_output.dim() == 3:  
            eye_output = eye_output[:, -1, :]  

        # Concatenate the outputs of both Transformers
        concatenated_output = torch.cat((eeg_output, eye_output), dim=1) 
        final_output = self.fc(concatenated_output)

        return final_output
