import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from optrans.datasets import adni
from optrans.continuous import RadonCDT

# X, y = adni.load_data()
X, y = adni.load_rcdt_maps()
img0 = adni.load_img0()

n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

n_components = 5

pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X)

components = pca.components_.reshape((n_components,h,w))
mean = pca.mean_.reshape((h,w))

print(X_pca.shape, pca.components_.shape, components.shape)

fig, ax = plt.subplots(1,5)
for i,a in enumerate(ax):
    a.imshow(components[i])

comp = 1
n_std = 1.5
n_steps = 5

radoncdt = RadonCDT()

std = X_pca[:,comp].std()
b = np.linspace(-n_std*std, n_std*std, n_steps)

fig, ax = plt.subplots(1, n_steps)
for i,a in enumerate(ax):
    X_recon = mean + components[comp]*b[i]

    img_recon = radoncdt.apply_inverse_map(X_recon, img0)
    a.imshow(img_recon)

plt.show()
