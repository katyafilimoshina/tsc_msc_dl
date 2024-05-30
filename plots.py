import numpy as np
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def visualize_dist_(points, ax, s=1000, alpha=1.0, cmap=None):
    """
    Visualizes distribution using sampled points
    """
    # Generate points from distribution
    X = points[:, 0]
    Y = points[:, 1]

    # Calculate the point density
    XY = torch.stack([torch.tensor(X), torch.tensor(Y)], dim=0).numpy()
    Z = gaussian_kde(XY)(XY)

    # Sort the points by density, so that the densest points are plotted last
    idx = Z.argsort()
    X, Y, Z = X[idx], Y[idx], Z[idx]

    ax.scatter(X, Y, c=Z, label=Z, alpha=alpha, cmap=cmap)