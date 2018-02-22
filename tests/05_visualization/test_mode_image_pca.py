import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from optrans.datasets import adni
from optrans.visualization import plot_mode_image

"""
Perform PCA on image data, and plot a PCA mode of variation as a single image.
"""

# Load some image data
X, y = adni.load_data()

# Reshape data to be n_imgs-by-n_features
n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

# Fit PCA transform
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Plot mode of variation image
ax = plot_mode_image([pca], component=0, shape=(h,w), n_std=3, n_steps=5)
plt.show()
