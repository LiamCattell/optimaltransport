import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import VOT2D
from optrans.datasets import adni
from optrans.utils import interp2d

X, _ = adni.load_data('../optrans/datasets/adni_data.npz')
img0 = adni.load_img0('../optrans/datasets/adni_img0.npy')
img1 = X[1]

h, w = img0.shape
x, y = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

vot = VOT2D(lr=0.01, alpha=0., max_iter=300, verbose=2)

img1_hat = vot.forward(img0, img1)
img0_recon = vot.apply_forward_map(vot.transport_map_, img1)
img1_recon = vot.apply_inverse_map(vot.transport_map_, img0)

fig, ax = plt.subplots(1, 3, sharex=True)
ax[0].plot(vot.cost_)
ax[0].set_title('Cost')
ax[1].plot(vot.mse_)
ax[1].set_title('MSE')
ax[2].plot(vot.curl_)
ax[2].set_title('Curl')

fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(img0)
ax[0,1].imshow(img1)
ax[1,0].imshow(img0_recon)
ax[1,1].imshow(img1_recon)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(vot.displacements_[0])
ax[1].imshow(vot.displacements_[1])
plt.show()
