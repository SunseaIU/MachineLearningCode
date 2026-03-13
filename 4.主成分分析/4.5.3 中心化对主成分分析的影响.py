import numpy as np
from scipy import linalg

# Define scales array
scales = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

# Set random seed for reproducibility
np.random.seed(0)
print(f"scale        corr1        corr2")
for scale in scales:
    # Define average array
    avg = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Generate 5000 examples, each 10-dimensional
    data = np.random.randn(5000, 10) + np.tile(avg * scale, (5000, 1))

    # Calculate mean and normalized mean
    m = np.mean(data, axis=0)
    m1 = m / np.linalg.norm(m)

    # PCA without centering
    _, S, V = linalg.svd(data)
    S = np.diag(S)
    e1 = V[:, 0]  # First eigenvector

    # PCA with centering
    newdata = data - np.tile(m, (5000, 1))
    U, S, V = linalg.svd(newdata)
    S = np.diag(S)
    new_e1 = V[:, 0]  # First eigenvector with centering

    # Calculate correlations
    avg = avg - np.mean(avg)
    avg = avg / np.linalg.norm(avg)

    e1 = e1 - np.mean(e1)
    e1 = e1 / np.linalg.norm(e1)

    new_e1 = new_e1 - np.mean(new_e1)
    new_e1 = new_e1 / np.linalg.norm(new_e1)

    corr1 = np.dot(avg, e1)
    corr2 = np.dot(e1, new_e1)

    # Print results
    print(f"{scale:.4f}     {corr1:.6f}     {corr2:.6f}")