import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import os

# Function to plot 2D data
def plot_2d_data(data, labels, title, save_path):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

# Load data for a single subject from .npy files
def load_single_subject_data(subject_id):
    data_dir = f'../dataset/{subject_id}/'
    X_train = np.load(os.path.join(data_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_label.npy'))
    return X_train, y_train

# Load data for all subjects combined from .npy files
def load_all_subjects_data():
    all_data = []
    all_labels = []
    for subject_id in [1, 2, 3]:
        data_dir = f'../dataset/{subject_id}/'
        X_train = np.load(os.path.join(data_dir, 'train_data.npy'))
        y_train = np.load(os.path.join(data_dir, 'train_label.npy'))
        all_data.append(X_train)
        all_labels.append(y_train)
    X_all = np.vstack(all_data)  # Combine all subject data
    y_all = np.hstack(all_labels)  # Combine all subject labels
    return X_all, y_all

# Perform PCA, LDA, and TSNE on the data and plot the results
def perform_dimensionality_reduction_and_plot(X, y, title_prefix, save_dir):
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plot_2d_data(X_pca, y, f'{title_prefix} PCA Dimensionality Reduction', 
                 save_path=os.path.join(save_dir, f'{title_prefix}_PCA.png'))

    # LDA
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)
    plot_2d_data(X_lda, y, f'{title_prefix} LDA Dimensionality Reduction', 
                 save_path=os.path.join(save_dir, f'{title_prefix}_LDA.png'))

    # TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plot_2d_data(X_tsne, y, f'{title_prefix} t-SNE Dimensionality Reduction', 
                 save_path=os.path.join(save_dir, f'{title_prefix}_TSNE.png'))

# Main function
if __name__ == "__main__":
    # Ensure save directory exists
    save_dir = '../assets'
    os.makedirs(save_dir, exist_ok=True)

    # Part 1: Perform dimensionality reduction on data for a single subject
    print("Performing dimensionality reduction for a single subject...")
    X_single, y_single = load_single_subject_data(1)  # Example: subject 1
    perform_dimensionality_reduction_and_plot(X_single, y_single, 'Subject_1', save_dir)

    # Part 2: Perform dimensionality reduction on combined data from all subjects
    print("Performing dimensionality reduction for all subjects combined...")
    X_all, y_all = load_all_subjects_data()
    perform_dimensionality_reduction_and_plot(X_all, y_all, 'All_Subjects', save_dir)