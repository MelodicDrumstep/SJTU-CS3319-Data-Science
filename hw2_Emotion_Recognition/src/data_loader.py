import os
import numpy as np

def load_seed_data_npy(data_path):
    """
    Load SEED dataset from .npy files for cross-subject validation
    :param data_path: Path to the dataset
    :return: List of (data, labels) for each subject
    """
    subject_data = []
    # Iterate through each subject folder
    for subject_folder in sorted(os.listdir(data_path)):
        subject_path = os.path.join(data_path, subject_folder)
        if os.path.isdir(subject_path):
            # Load data.npy and label.npy for each subject
            data_file = os.path.join(subject_path, 'data.npy')
            label_file = os.path.join(subject_path, 'label.npy')
            if os.path.exists(data_file) and os.path.exists(label_file):
                subject_data_array = np.load(data_file)
                subject_labels_array = np.load(label_file)
                # Append (data, labels) tuple for each subject
                subject_data.append((subject_data_array, subject_labels_array))
    return subject_data


def reshape_to_spatial(data, channel_matrix, channelID2str):
    """
    Reshape function for CNN. Currently it's not in use.
    """
    reshaped_data = np.zeros((data.shape[0], 8, 9, 5))  # (samples, height, width, frequency bands)
    channel_map = {v: k-1 for k, v in channelID2str.items()}  # Map channel names to indices
    
    for i in range(8):
        for j in range(9):
            channel_name = channel_matrix[i][j]
            if channel_name != '-':
                channel_idx = channel_map[channel_name]
                reshaped_data[:, i, j, :] = data[:, channel_idx, :]
    
    return reshaped_data
