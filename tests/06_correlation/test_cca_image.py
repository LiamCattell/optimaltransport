import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from optrans.datasets import oasis
from optrans.decomposition import CCA, fit_line
from optrans.continuous import RadonCDT
from optrans.visualization import plot_mode_image

# X, y, metadata = oasis.load_data()
X, y, metadata = oasis.load_rcdt_maps()
img0 = oasis.load_img0()

ind = ~np.isnan(metadata).any(axis=1)
X = X[ind]
y = y[ind]
m = metadata[ind,1]

n_samples, h, w = X.shape
X = X.reshape((n_samples,h*w))

Xtr, Xte, ytr, yte, mtr, mte = train_test_split(X, y, m, test_size=0.5,
                                                random_state=43)

pca = PCA(n_components=10)
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)

cca = CCA(n_components=1)
cca.fit(Xtr_pca, mtr.reshape(-1,1))

Xtr_cca = cca.transform(Xtr_pca)
Xte_cca = cca.transform(Xte_pca)

score_tr = cca.score(Xtr_pca, mtr.reshape(-1,1))
score_te = cca.score(Xte_pca, mte.reshape(-1,1))
print('Score tr: ', score_tr)
print('Score te: ', score_te)


plot_mode_image([pca,cca], shape=(h,w), transform=RadonCDT(), img0=img0, n_std=7.)
plt.show()


std = np.sqrt(cca.explained_variance_[0])
std_range = np.linspace(-1.5, 1.5, 5)
img_recon = []
radoncdt = RadonCDT()


for sr in std_range:
    X_pca_recon = cca.inverse_transform(np.array([[std*sr]]))
    X_recon = pca.inverse_transform(X_pca_recon)
    f_recon = X_recon.reshape((h,w))
    img_recon.append(radoncdt.apply_inverse_map(f_recon, img0))

fig, ax = plt.subplots(1, std_range.size)
for im,a in zip(img_recon,ax):
    a.imshow(im, cmap='gray')
plt.show()

plt.imshow(img_recon[0]-img_recon[-1])
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
