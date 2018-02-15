import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import RadonCDT
from optrans.utils import signal_to_pdf
from optrans.datasets import adni, gaussians

X, y = gaussians.load_data()

img0 = X[0]
img1 = X[35]

radoncdt = RadonCDT()

rcdt = radoncdt.forward(img0, img1)
x = radoncdt.transport_map_ + radoncdt.displacements_

vmin = min(img0.min(), img1.min())
vmax = max(img0.max(), img1.max())

fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
for i,alpha in enumerate(np.linspace(0, 1, 5)):
    # Interpolation in image space
    img_interp = (1. - alpha) * img0 + alpha * img1
    ax[0,i].imshow(img_interp)
    ax[0,i].set_title('alpha = {:.2f}'.format(alpha))

    # Interpolation in Radon-CDT space
    u = radoncdt.displacements_ * alpha
    f = x - u
    img_recon = radoncdt.apply_inverse_map(f, img0)
    ax[1,i].imshow(img_recon, vmin=vmin, vmax=vmax)

ax[0,0].set_ylabel('Image space')
ax[1,0].set_ylabel('CDT space')
plt.show()
