import os
import numpy as np

def load_subject_data(data_path):
    """
    Load SEED dataset from .npy files for each subject's train and test data.
    :param data_path: Path to the dataset directory
    :return: Dictionary with subject data, where key is subject id and value is (train_data, test_data, labels)
    """
    subject_data = {}
    
    for subject_folder in sorted(os.listdir(data_path)):
        subject_path = os.path.join(data_path, subject_folder)
        if os.path.isdir(subject_path):
            train_data_eeg_file = os.path.join(subject_path, 'train_data_eeg.npy')
            train_data_eye_file = os.path.join(subject_path, 'train_data_eye.npy')
            train_labels_file = os.path.join(subject_path, 'train_label.npy')
            
            test_data_eeg_file = os.path.join(subject_path, 'test_data_eeg.npy')
            test_data_eye_file = os.path.join(subject_path, 'test_data_eye.npy')
            test_labels_file = os.path.join(subject_path, 'test_label.npy')
            
            if all(os.path.exists(f) for f in [train_data_eeg_file, train_data_eye_file, train_labels_file,
                                               test_data_eeg_file, test_data_eye_file, test_labels_file]):
                train_data_eeg = np.load(train_data_eeg_file) 
                train_data_eye = np.load(train_data_eye_file) 
                train_labels = np.load(train_labels_file)     
                
                test_data_eeg = np.load(test_data_eeg_file)   
                test_data_eye = np.load(test_data_eye_file)   
                test_labels = np.load(test_labels_file)       
                
                subject_data[int(subject_folder)] = {
                    'train_data_eeg': train_data_eeg,
                    'train_data_eye': train_data_eye,
                    'train_labels': train_labels,
                    'test_data_eeg': test_data_eeg,
                    'test_data_eye': test_data_eye,
                    'test_labels': test_labels
                }
    
    return subject_data
