"""
helpers
"""
import numpy as np


def gaussian_matrix(rows, cols, sigma=1.0):
    """
    Generate a 2D Gaussian matrix of size (rows, cols) with standard deviation sigma.
    The Gaussian is centered in the middle of the matrix.
    """
    # Create coordinate grids
    x = np.linspace(-(cols - 1) / 2., (cols - 1) / 2., cols)
    y = np.linspace(-(rows - 1) / 2., (rows - 1) / 2., rows)
    xx, yy = np.meshgrid(x, y)

    # 2D Gaussian formula
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    # Normalize so that sum = 1
    return g / g.sum()