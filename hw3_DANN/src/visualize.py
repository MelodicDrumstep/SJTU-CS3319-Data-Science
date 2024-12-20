import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import argparse

def visualize_features(feature_file, title_str):
    """
    Visualize features using t-SNE.
    :param feature_file: Path to the file containing features.
    """
    data = np.load(feature_file)
    features = data['features']
    labels = data['labels']

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=f"Class {label}", alpha=0.6)
    plt.legend()
    plt.title(f"t-SNE Visualization of Extracted Features from {title_str}")
    plt.savefig(f"images/{title_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', type=str, required=True, help='feature file path')
    parser.add_argument('--age', type=int, default=18, help='年龄')
    parser.add_argument('--verbose', action='store_true', help='是否启用详细模式')

    args = parser.parse_args()

    visualize_features(args.filepath, "DANN-DG with domain classifier")