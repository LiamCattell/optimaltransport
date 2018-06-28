import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from optrans.datasets import oasis
from optrans.decomposition import CCA, fit_line
from optrans.visualization import plot_mode_image

"""
Perform CCA on the OASIS brain MRI dataset. Find correlation between the images
and metadata. Plots the data projected on to the first canonical direction.

Liam Cattell -- May 2018
"""

# Load data
X, y, metadata = oasis.load_data()

# Remove NaNs in the metadata
ind = ~np.isnan(metadata).any(axis=1)
X = X[ind]
y = y[ind]
m = metadata[ind,1]  # Age

# Reshape data into n-by-p array
n_samples, h, w = X.shape
X = X.reshape((n_samples,h*w))

# Split data into training and testing sets
Xtr, Xte, ytr, yte, mtr, mte = train_test_split(X, y, m, test_size=0.25,
                                                random_state=42)


# Perform PCA before CCA
pca = PCA(n_components=10)
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)

# CCA
cca = CCA(n_components=1)
cca.fit(Xtr_pca, mtr.reshape(-1,1))
Xtr_cca = cca.transform(Xtr_pca)
Xte_cca = cca.transform(Xte_pca)

# Get the correlation coefficients of the training and testing data
score_tr = cca.score(Xtr_pca, mtr.reshape(-1,1))
score_te = cca.score(Xte_pca, mte.reshape(-1,1))
print('Score train: ', score_tr)
print('Score test:  ', score_te)

# Plot the mode of variation along the first CCA direction
plot_mode_image([pca,cca], shape=(h,w), transform=None, n_std=7.)
plt.show()

# Get x-y coordinates of correlation line
xl, yl = fit_line(Xte_cca.squeeze(), mte)

# Plot data projected on to first canonical direction
plt.scatter(Xtr_cca, mtr, c='b', label='train')
plt.scatter(Xte_cca, mte, c='r', label='test')
plt.plot(xl, yl, 'k--', lw=2, label='corr. test')
plt.xlabel('X scores')
plt.ylabel('m scores')
plt.title('R_tr = {:.2f}\nR_te = {:.2f}'.format(score_tr, score_te))
plt.legend()
plt.show()
