"""Projection 2D par ACP manuelle (calcul explicite de la matrice de
covariance et de ses vecteurs propres), utilisée pour visualiser l'effet
de la réduction du dataset (src/reduction.py)."""
import matplotlib.pyplot as plt
import numpy as np


def plot_pca_2d(X, y, titre):
    """Projette (X, y) sur ses 2 premières composantes principales et
    affiche un scatter coloré par label."""
    X_pca = np.concatenate((X, y.reshape((y.shape[0], 1))), axis=1)

    labels = X_pca[:, -1].astype(int)
    X_features = X_pca[:, :-1]

    X_meaned = X_features - np.mean(X_features, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:2]
    X_reduced = np.dot(X_meaned, eigenvector_subset)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.legend(*scatter.legend_elements(), title="Labels")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(titre)
    plt.grid(True)
    plt.show()
