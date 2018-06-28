import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from optrans.datasets import adni
from optrans.visualization import plot_mode_image
from optrans.continuous import RadonCDT

"""
Perform PCA on Radon-CDT data, and plot a PCA mode of variation (in image
space) as a single image.

Liam Cattell -- March 2018
"""

# Load some Radon-CDT maps
X, y = adni.load_rcdt_maps()
img0 = adni.load_img0()

# Reshape data to be n_imgs-by-n_features
n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

# Fit PCA transform
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Plot mode of variation image
ax = plot_mode_image([pca], component=0, shape=(h,w), transform=RadonCDT(),
                     img0=img0, n_std=3, n_steps=5)
plt.show()
