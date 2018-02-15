import numpy as np
from skimage.transform import radon, iradon

from .base import BaseTransform
from .cdt import CDT
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf, match_shape2d


class RadonCDT(BaseTransform):
    """
    Radon Cumulative Distribution Transform.

    Parameters
    ----------
    theta : 1d array (default=np.arange(180))
        Radon transform projection angles.

    Attributes
    -----------
    displacements_ : array, shape (t, len(theta))
        Displacements u.
    transport_map_ : array, shape (t, len(theta))
        Transport map f.
    """
    def __init__(self, theta=np.arange(180)):
        super(RadonCDT, self).__init__()
        self.theta = check_array(theta, ndim=1)
        self.epsilon = 1e-8
        self.total = 1.


    def forward(self, sig0, sig1):
        """
        Forward transform.

        Parameters
        ----------
        sig0 : array, shape (height, width)
            Reference image.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        rcdt : array, shape (t, len(theta))
            Radon-CDT of input image sig1.
        """
        # Check input arrays
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Input signals must be the same size
        assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        # Set reference signal
        self.sig0_ = sig0

        # Radon transform of signals
        rad0 = radon(sig0, theta=self.theta, circle=False)
        rad1 = radon(sig1, theta=self.theta, circle=False)

        # Initalize CDT, Radon-CDT, displacements, and transport map
        cdt = CDT()
        rcdt = []
        u = []
        f = []

        # Loop over angles
        for i in range(self.theta.size):
            # Convert projection to PDF
            j0 = signal_to_pdf(rad0[:,i], epsilon=self.epsilon,
                               total=self.total)
            j1 = signal_to_pdf(rad1[:,i], epsilon=self.epsilon,
                               total=self.total)

            # Compute CDT of this projection
            lot = cdt.forward(j0, j1)

            # Update 2D Radon-CDT, displacements, and transport map
            rcdt.append(lot)
            u.append(cdt.displacements_)
            f.append(cdt.transport_map_)

        # Convert lists to arrays
        rcdt = np.asarray(rcdt).T
        self.displacements_ = np.asarray(u).T
        self.transport_map_ = np.asarray(f).T

        self.is_fitted = True

        return rcdt


    def inverse(self):
        """
        Inverse transform.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        self._check_is_fitted()
        return self.apply_inverse_map(self.transport_map_, self.sig0_)


    def apply_forward_map(self, transport_map, sig1):
        """
        Appy forward transport map.

        Parameters
        ----------
        transport_map : array, shape (t, len(theta))
            Forward transport map.
        sig1 : 2d array, shape (height, width)
            Signal to transform.
        theta : array (default=np.arange(180))
            Radon transform projection angles.

        Returns
        -------
        sig0_recon : array, shape (height, width)
            Reconstructed reference signal sig0.
        """
        # Check input arrays
        transport_map = check_array(transport_map, ndim=2,
                                    dtype=[np.float64, np.float32])
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Number of projections in transport map must match number of angles
        if transport_map.shape[1] != self.theta.size:
            raise ValueError("Length of theta must equal number of "
                             "projections in transport map: {} vs "
                             "{}".format(self.theta.size, transport_map.shape[1]))

        # Initialize Radon transforms
        rad1 = radon(sig1, theta=self.theta, circle=False)
        rad0 = np.zeros_like(rad1)

        # Check transport map and Radon transforms are the same size
        assert_equal_shape(transport_map, rad0,
                           ['transport_map', 'Radon transform of sig0'])

        # Loop over angles
        cdt = CDT()
        for i in range(self.theta.size):
            # Convert projection to PDF
            j1 = signal_to_pdf(rad1[:,i], epsilon=1e-8, total=1.)

            # Radon transform of sig0 comprised of inverse CDT of projections
            rad0[:,i] = cdt.apply_forward_map(transport_map[:,i], j1)

        # Inverse Radon transform
        sig0_recon = iradon(rad0, self.theta, circle=False, filter='ramp')

        # Crop sig0_recon to match sig1
        sig0_recon = match_shape2d(sig1, sig0_recon)

        return sig0_recon


    def apply_inverse_map(self, transport_map, sig0):
        """
        Appy inverse transport map.

        Parameters
        ----------
        transport_map : 2d array, shape (t, len(theta))
            Forward transport map. Inverse is computed in this function.
        sig0 : array, shape (height, width)
            Reference signal.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        # Check input arrays
        transport_map = check_array(transport_map, ndim=2,
                                    dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Initialize Radon transforms
        rad0 = radon(sig0, theta=self.theta, circle=False)
        rad1 = np.zeros_like(rad0)

        # Check transport map and Radon transforms are the same size
        assert_equal_shape(transport_map, rad0,
                           ['transport_map', 'Radon transform of sig0'])

        # Loop over angles
        cdt = CDT()
        for i in range(self.theta.size):
            # Convert projection to PDF
            j0 = signal_to_pdf(rad0[:,i], epsilon=1e-8, total=1.)

            # Radon transform of sig1 comprised of inverse CDT of projections
            rad1[:,i] = cdt.apply_inverse_map(transport_map[:,i], j0)

        # Inverse Radon transform
        sig1_recon = iradon(rad1, self.theta, circle=False, filter='ramp')

        # Crop sig1_recon to match sig0
        sig1_recon = match_shape2d(sig0, sig1_recon)

        return sig1_recon
