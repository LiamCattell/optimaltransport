import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from optrans.datasets import gaussians
from optrans.decomposition import PLDA

"""
Classify Gaussian blobs using PCA + PLDA.
"""

# Load data
X, y = gaussians.load_rcdt()
# X, y = gaussians.load_data()

# Reshape data into a n-by-p 2d array
n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

# Split data into training and test sets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=11,
                                      stratify=y)

# Perform PCA to reduce dimensionality before classification
pca = PCA()
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)

# Train PLDA classifier
plda = PLDA(n_components=2, alpha=1.)
plda.fit(Xtr_pca, ytr)

# Test classifier
Xte_plda = plda.transform(Xte_pca)
y_pred = plda.predict(Xte_pca)
acc = accuracy_score(yte, y_pred)
print("ACC: {:.3f}".format(acc))

# Create a grid of points to plot the shaded regions
xx, yy = np.meshgrid(np.linspace(Xte_plda[:,0].min(),Xte_plda[:,0].max(),200),
                     np.linspace(Xte_plda[:,1].min(),Xte_plda[:,1].max(),200))

# Transform grid of points into PCA space, so that they can be used as inputs
# to the PLDA classifier
grid_pca = plda.mean_ + (np.c_[xx.ravel(),yy.ravel()]).dot(plda.components_)

# Get predicted classes of grid points
zz = plda.predict(grid_pca).reshape(xx.shape)

# Plot test data
fig, ax = plt.subplots(1, 1)
ax.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.6)
ax.scatter(Xte_plda[:,0], Xte_plda[:,1], c=yte, cmap=plt.cm.coolwarm, s=20,
           edgecolors='k', alpha=0.8)
plt.show()
