import numpy as np
import matplotlib.pyplot as plt

from optrans.datasets import adni
from optrans.decomposition import PLDA

ind = 211

X, y = adni.load_data()

n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

plda = PLDA(alpha=1.)
X_plda = plda.fit_transform(X, y)
X_recon = plda.mean_ + X_plda[ind].reshape(1,-1).dot(plda.components_)

mean = plda.mean_.reshape((h,w))
img = X[ind].reshape((h,w))
img_recon = X_recon.reshape((h,w))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title("Original")
ax[1].imshow(img_recon)
ax[1].set_title("Recon.")
plt.show()
