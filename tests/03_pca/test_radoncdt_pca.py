import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from optrans.datasets import adni, gaussians
from optrans.continuous import RadonCDT
from optrans.decomposition import get_mode_variation, get_mode_histogram

"""
Perform PCA on the Radon-CDTs of some Gaussian 'blob' images.

In image space, PCA is unable to capture the non-linearities (translations) in
the dataset. However, in transport space, the Radon-CDT can account for the
non-linearity, and thus, PCA is able to reconstruct the data from only two
components.
"""

# Load some Radon-CDT data
X, y = gaussians.load_rcdt_maps()
img0 = gaussians.load_img0()
X = X[y==0]
y = y[y==0]

# Reshape data to be n_imgs-by-n_features
n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

# Fit PCA transform
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Plot the first 5 PCA components
fig0, ax0 = plt.subplots(1,5)
for i,a in enumerate(ax0):
    component = pca.components_[i].reshape((h,w))
    a.imshow(component)
    a.set_title("Component {}".format(i))

# Plot data projected along first two components
fig1, ax1 = plt.subplots(1,1)
ax1.scatter(X_pca[:,0], X_pca[:,1])
ax1.set_xlabel('1st component')
ax1.set_ylabel('2nd component')

# Get the mode of variation along the first component
ind = 0
n_std = 2.
n_steps = 5
mode = get_mode_variation(pca, component=ind, n_std=n_std, n_steps=n_steps)

# Initialise Radon-CDT so we can compute inverse transform
radoncdt = RadonCDT()

# Get std dev. along component
std = np.sqrt(pca.explained_variance_[ind])
std_range = np.linspace(-n_std/2, n_std/2, n_steps)

# Plot mode of variation
fig2, ax2 = plt.subplots(1, n_steps)
for m,s,a in zip(mode,std_range,ax2):
    img_recon = radoncdt.apply_inverse_map(m.reshape((h,w)), img0)
    a.imshow(img_recon)
    a.set_title("{:.2f}$\sigma$".format(s))

# Get histogram of data projected on to component
hist, bin_centers = get_mode_histogram(X_pca/std, y, component=ind, bins=7,
                                       range=(std_range[0],std_range[-1]))
wid = (bin_centers[0]-bin_centers[1]) / 2

# Plot histogram
fig3, ax3 = plt.subplots(1,1)
ax3.bar(bin_centers, hist, width=wid)
ax3.set_title('Data projected on to component {}'.format(ind))
ax3.set_xlabel('$\sigma$')

plt.show()
