import torch
import torch.nn as nn

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

        # EEG Transformer
        self.eeg_transformer = nn.Transformer(
            d_model=hidden_size, 
            num_encoder_layers=num_layers, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4
        )
        
        # Eye Transformer
        self.eye_transformer = nn.Transformer(
            d_model=hidden_size, 
            num_encoder_layers=num_layers, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4
        )

        # Fully connected layer after concatenation of outputs from both Transformers
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Hidden size * 2 after concatenation

    def forward(self, eeg_data, eye_data):
        """
        Forward pass through the Decision-Level Fusion Transformer
        :param eeg_data: EEG data (batch_size, seq_len, input_size_eeg)
        :param eye_data: Eye tracking data (batch_size, seq_len, input_size_eye)
        :return: The fused output (final prediction)
        """
        # Pass EEG data through EEG Transformer
        eeg_output = self.eeg_transformer(eeg_data)

        # Pass Eye tracking data through Eye Transformer
        eye_output = self.eye_transformer(eye_data)

        # Take the output from the last time step (assuming seq_len=1)
        eeg_output = eeg_output[:, -1, :]  # (batch_size, hidden_size)
        eye_output = eye_output[:, -1, :]  # (batch_size, hidden_size)

        # Concatenate the outputs of both Transformers
        concatenated_output = torch.cat((eeg_output, eye_output), dim=1)  # (batch_size, hidden_size * 2)

        # Pass the concatenated output through the fully connected layer
        final_output = self.fc(concatenated_output)

        return final_output
