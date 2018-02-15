import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import RadonCDT
from optrans.utils import signal_to_pdf

sigma = 6
n = 128

img0 = np.zeros((n,n))
img1 = np.zeros((n,n))

img0[96,32] = 1.
img1[32,96] = 1.

img0 = signal_to_pdf(img0, sigma=sigma)
img1 = signal_to_pdf(img1, sigma=sigma)

radoncdt = RadonCDT()

rcdt = radoncdt.forward(img0, img1)
x = radoncdt.transport_map_ + radoncdt.displacements_

fig, ax = plt.subplots(2, 5)
for i,alpha in enumerate(np.linspace(0, 1, 5)):
    # Interpolation in image space
    img_interp = (1. - alpha) * img0 + alpha * img1
    ax[0,i].imshow(img_interp)
    ax[0,i].set_title('alpha = {:.2f}'.format(alpha))

    # Interpolation in Radon-CDT space
    u = radoncdt.displacements_ * alpha
    f = x - u
    img_recon = radoncdt.apply_inverse_map(f, img0)
    ax[1,i].imshow(img_recon)

ax[0,0].set_ylabel('Image space')
ax[1,0].set_ylabel('Radon-CDT space')
plt.show()
