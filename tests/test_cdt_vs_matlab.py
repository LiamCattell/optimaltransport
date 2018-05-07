import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from scipy.io import loadmat

from optrans.continuous import CDT
from optrans.utils import signal_to_pdf


def sig2pdf(sig, epsilon, total):
    sig_out = total * sig / sig.sum()
    sig_out += epsilon
    sig_out = total * sig_out / sig_out.sum()
    return sig_out

epsilon = 1e-8
total = 1.

# img = camera().astype(np.float)
data = loadmat("C:/Users/lcc3d/Documents/MATLAB/cmu_code/ContinuousLOT/ContinuousLOT/Data/FaceSamples.mat")
img = data['I'][0][0].astype(np.float)

h, w = img.shape
# sig0 = sig2pdf(np.ones(w), epsilon, total)
sig0 = sig2pdf(img[36], epsilon, total)
sig1 = sig2pdf(img[10], epsilon, total)

# sig0 /= sig0.sum()
# sig1 /= sig1.sum()

cdt = CDT()
sig1_hat = cdt.forward(sig0, sig1)

sig0_recon = cdt.apply_forward_map(cdt.transport_map_, sig1)
sig1_recon = cdt.apply_inverse_map(cdt.transport_map_, sig0)

fig, ax = plt.subplots(3, 2)
ax[0,0].plot(sig0)
ax[0,1].plot(sig1)
ax[1,0].plot(sig0_recon)
ax[1,1].plot(sig1_recon)
ax[2,0].plot(cdt.transport_map_)
ax[2,1].plot(cdt.displacements_)
plt.show()
