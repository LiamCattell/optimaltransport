import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from optrans.datasets import gaussians
from optrans.decomposition import PLDA
from optrans.visualization import plot_mode_histogram
from optrans.continuous import RadonCDT

"""
Plot the histograms of the Gaussian blob data projected on to the 1st PLDA
direction.

Liam Cattell -- May 2018
"""

# Load some image data
X, y = gaussians.load_rcdt_maps()
img0 = gaussians.load_img0()

# Reshape data to be n_imgs-by-n_features
n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

# Fit PCA transform
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)

# Fit PLDA transform
plda = PLDA(n_components=5)
X_plda = plda.fit_transform(X_pca, y)

# Plot scatter
fig, ax0 = plt.subplots(1, 1)
for lab in np.unique(y):
    ax0.scatter(X_plda[y==lab,0], X_plda[y==lab,1])

# Plot the histogram of data projected on to the 1st component
ax1 = plot_mode_histogram(X_plda, y=y, component=0, decomp=plda, n_bins=11)

plt.show()
