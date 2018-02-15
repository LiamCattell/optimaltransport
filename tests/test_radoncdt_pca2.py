import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from optrans.datasets import gaussians
from optrans.continuous import RadonCDT


def run_pca(X, te, n_components):
    n_imgs, h, w = X.shape
    X = X.reshape((n_imgs,h*w))
    tr = np.ones(n_imgs, dtype=bool)
    tr[te] = False

    pca = PCA(n_components=n_components)

    pca.fit(X[tr])

    Xte_pca = pca.transform(X[te].reshape(1,-1))
    Xte_recon = pca.mean_ + Xte_pca.dot(pca.components_)
    Xte_recon = Xte_recon.reshape((h,w))

    return X[te].reshape((h,w)), Xte_recon


n_components = 5
te = 4

X_img, _ = gaussians.load_data()
X_rcdt, _ = gaussians.load_rcdt_maps()
img0 = gaussians.load_img0()

X_img = X_img[:100]
X_rcdt = X_rcdt[:100]

img, img_recon = run_pca(X_img, te, n_components)

rcdt, rcdt_recon = run_pca(X_rcdt, te, n_components)
rimg = RadonCDT.apply_inverse_map(rcdt, img0)
rimg_recon = RadonCDT.apply_inverse_map(rcdt_recon, img0)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(img)
ax[0,0].set_title("Test image")
ax[0,1].imshow(img_recon)
ax[0,1].set_title("Image recon.\n({} comp.)".format(n_components))
ax[1,0].imshow(rimg)
ax[1,0].set_title("Inv. R-CDT")
ax[1,1].imshow(rimg_recon)
ax[1,1].set_title("R-CDT recon.\n({} comp.)".format(n_components))
plt.show()
