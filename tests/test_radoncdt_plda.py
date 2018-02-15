import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from optrans.datasets import adni
from optrans.continuous import RadonCDT
from optrans.decomposition import PLDA

# X, y = adni.load_data()
X, y = adni.load_rcdt_maps()
img0 = adni.load_img0()

n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

alpha = 0.1
n_components = 5

plda = PLDA(alpha=alpha, n_components=n_components)

X_plda = plda.fit_transform(X, y)
print("ACC: ", plda.score(X, y))

components = plda.components_.reshape((n_components,h,w))
mean = plda.mean_.reshape((h,w))

print(X_plda.shape, plda.components_.shape, components.shape)

fig, ax = plt.subplots(1,5)
for i,a in enumerate(ax):
    a.imshow(components[i])

comp = 0
n_std = 2
n_steps = 5

radoncdt = RadonCDT()

std = X_plda[:,comp].std()
b = np.linspace(-n_std*std, n_std*std, n_steps)

# img_mean = mean
img_mean = radoncdt.apply_inverse_map(mean, img0)

fig, ax = plt.subplots(2, n_steps)
for i in range(n_steps):
    X_recon = mean + components[comp]*b[i]

    # img_recon = X_recon
    img_recon = radoncdt.apply_inverse_map(X_recon, img0)

    ax[0,i].imshow(img_recon)
    ax[1,i].imshow(img_recon - img_mean)

plt.show()
