import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Dimensionality disaster section
def generate_data_and_plot_histogram(dimensions, num_samples=100):
    for dim in dimensions:
        # Generate random samples from standard normal distribution
        data = np.random.randn(num_samples, dim)
        # Calculate pairwise distances between data points
        distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2).flatten()
        
        # Plot histogram of distances
        plt.hist(distances, bins=200)
        plt.title(f'Histogram of distances for dimension {dim}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        # Save the figure instead of showing it
        plt.savefig(f'../assets/histogram_dim_{dim}.png')
        # Close the figure to free up memory
        plt.close()

# Call the function for dimensionality disaster experiment
generate_data_and_plot_histogram([2, 100, 1000, 10000])
