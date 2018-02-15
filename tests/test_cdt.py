import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import img_as_float
from scipy.ndimage.filters import gaussian_filter

from optrans.utils import signal_to_pdf
from optrans.continuous import CDT

# img = img_as_float(data.camera()[::2,::2])
#
# sigma = 3
# sig0 = signal_to_pdf(img[75,:], sigma=sigma)
# sig1 = signal_to_pdf(img[150,:], sigma=sigma)

x = np.linspace(-5,5,256)
sig0 = np.ones(256)
sig1 = 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2)

sig0 = signal_to_pdf(sig0)
sig1 = signal_to_pdf(sig1)

cdt = CDT()

lot = cdt.forward(sig0, sig1)
sig0_recon = cdt.apply_forward_map(cdt.transport_map_, sig1)
sig1_recon = cdt.apply_inverse_map(cdt.transport_map_, sig0)
# sig1_recon = cdt.inverse(sig0)

fig, ax = plt.subplots(3, 2, sharex=True)
ax[0,0].plot(sig0)
ax[0,0].set_title('Reference sig0')
ax[0,1].plot(sig1)
ax[0,1].set_title('Signal sig1')
ax[1,0].plot(sig0_recon)
ax[1,0].set_title('Reconstructed sig0')
ax[1,1].plot(sig1_recon)
ax[1,1].set_title('Reconstructed sig1')
ax[2,0].plot(cdt.displacements_)
ax[2,0].set_title('Displacements u')
ax[2,1].plot(lot)
ax[2,1].set_title('CDT')

plt.show()
