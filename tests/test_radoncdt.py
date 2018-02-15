import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import img_as_float
from scipy.ndimage.filters import gaussian_filter

from optrans.utils import signal_to_pdf
from optrans.continuous import RadonCDT

img1 = img_as_float(data.camera()[::2,::2])
img0 = img1[50:162,70:134]
img1 = img1[32:144,64:128]
# img0 = np.ones_like(img1)

sigma = 1
sig0 = signal_to_pdf(img0, sigma=sigma)
sig1 = signal_to_pdf(img1, sigma=sigma)

radoncdt = RadonCDT()

rcdt = radoncdt.forward(sig0, sig1)
sig0_recon = radoncdt.apply_forward_map(radoncdt.transport_map_, sig1)
sig1_recon = radoncdt.inverse(sig0)


fig, ax = plt.subplots(3,2)
ax[0,0].imshow(sig0)
ax[0,0].set_title('Reference sig0')
ax[0,1].imshow(sig1)
ax[0,1].set_title('Signal sig1')
ax[1,0].imshow(sig0_recon)
ax[1,0].set_title('Reconstructed sig0')
ax[1,1].imshow(sig1_recon)
ax[1,1].set_title('Reconstructed sig1')
ax[2,0].imshow(radoncdt.displacements_)
ax[2,0].set_title('Displacements u')
ax[2,1].imshow(rcdt)
ax[2,1].set_title('Radon-CDT')

plt.show()
