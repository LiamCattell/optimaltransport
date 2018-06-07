import numpy as np
from skimage.transform import pyramid_reduce, pyramid_expand, resize

from .base import BaseTransform
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf, interp2d, griddata2d


class VOT2D(BaseTransform):
    """
    Variational Optimal Transport 2D Transform.

    .. note::

    Unlike the original paper by Kundu et al., this implementation uses the
    Adam gradient descent optimizer.

    Parameters
    ----------
    alpha : float (default=0.01)
        Regularization of the curl operator.
    lr : float (default=0.01)
        Learning rate.
    beta1 : float (default=0.9)
        Adam optimizer parameter. 0 < beta1 < 1. Generally close to 1.
    beta2 : float (default=0.999)
        Adam optimizer parameter. 0 < beta2 < 1. Generally close to 1.
    decay : float (default=0.)
        Learning rate decay over each update.
    max_iter : int (default=300)
        Maximum number of iterations.
    tol : float (default=0.001)
        Stop iterating when change in cost function is below this threshold.
    verbose : int (default=1)
        Verbosity during optimization. 0=no output, 1=print cost,
        2=print all metrics.

    Attributes
    -----------
    displacements_ : array, shape (2, height, width)
        Displacements u. First index denotes direction: displacements_[0] is
        y-displacements, and displacements_[1] is x-displacements.
    transport_map_ : array, shape (2, height, width)
        Transport map f. First index denotes direction: transport_map_[0] is
        y-map, and transport_map_[1] is x-map.
    cost_ : list of float
        Value of cost function at each iteration.
    mse_ : list of float
        Mean squared error at each iteration.
    curl_ : list of float
        Curl at each iteration.

    References
    ----------
    [Discovery and visualization of structural biomarkers from MRI using
    transport-based morphometry]
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5912801/)
    [Adam - A method for stochastic optimization]
    (http://arxiv.org/abs/1412.6980v8)

    .. seealso::

    MultiVOT2D
    """
    def __init__(self, alpha=0.01, lr=0.01, beta1=0.9, beta2=0.999, decay=0.,
                 max_iter=300, tol=0.001, verbose=0):
        super(VOT2D, self).__init__()
        self.alpha = alpha
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        return


    def forward(self, sig0, sig1, f_init=None):
        """
        Forward transform.

        Parameters
        ----------
        sig0 : array, shape (height, width)
            Reference image.
        sig1 : array, shape (height, width)
            Signal to transform.
        f_init : None or array, shape (2, height, width)
            Initial transport map. If None, the initial transport map is the
            identity transform.

        Returns
        -------
        lot : array, shape (2, height, width)
            LOT transform of input image sig1. First index denotes direction:
            lot[0] is y-LOT, and lot[1] is x-LOT.
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

        # Initialise regular grid
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

        # If no initial f is provided, set transport map to be identity
        # transform (i.e. f = x)
        if f_init is None:
            f = np.stack((yv,xv), axis=0)
        else:
            f = np.copy(f_init)

        # Set the fill value for interpolation
        fill_val = min(sig0.min(), sig1.min())

        # Create a mask to avoid edge effects
        mask = np.zeros_like(sig0)
        mask[2:-2,2:-2] = 1.

        # Initialise evaluation measures
        self.cost_ = []
        self.mse_ = []
        self.curl_ = []

        # Initialise derivative of cost function wrt f
        ft = np.zeros_like(f)

        # Initialise Adam moment estimates
        mt = np.zeros_like(f)
        vt = np.zeros_like(f)

        # Iterate!
        for i in range(self.max_iter):
            # Jacobian and its determinant
            f0y, f0x = np.gradient(f[0])
            f1y, f1x = np.gradient(f[1])
            detJ = (f1x * f0y) - (f1y * f0x)

            # Transform sig1 using f (i.e. sig1(f))
            sig1f = interp2d(sig1, f, fill_value=fill_val)
            sig1fy, sig1fx = np.gradient(sig1f)

            # Reconstructed sig0 and its error
            sig0_recon = detJ * sig1f
            err = sig0_recon - sig0

            # Update evaluation metrics
            cost = 0.5*np.sum(err**2) + self.alpha*np.sum((f0x-f1y)**2)
            curl = 0.5 * (f0x - f1y)
            self.cost_.append(cost)
            self.mse_.append(np.mean(err**2))
            self.curl_.append(0.5*np.sum(curl**2))

            # Print cost value
            if self.verbose:
                print('Iteration {:>4} -- '
                      'cost = {:.4e}'.format(i, self.cost_[-1]))

            # Print MSE and curl
            if self.verbose > 1:
                print('... mse = {:.4e}, '
                      'curl = {:.4e}'.format(self.mse_[-1], self.curl_[-1]))

            # Useful 2nd derivatives
            f0xy, f0xx = np.gradient(f0x)
            f1yy, f1yx = np.gradient(f1y)

            # Compute divergence
            _, g0x = np.gradient(-f1y*err*sig1f)
            g0y, _ = np.gradient(f1x*err*sig1f)
            _, g1x = np.gradient(f0y*err*sig1f)
            g1y, _ = np.gradient(-f0x*err*sig1f)
            div0 = g0x + g0y
            div1 = g1x + g1y

            # Derivative of cost function wrt f
            ft[0] = detJ * sig1fy * err - div0 + self.alpha*(f1yx-f0xx)
            ft[1] = detJ * sig1fx * err - div1 + self.alpha*(f0xy-f1yy)

            # Mask the derivative to avoid edge effects
            ft *= mask

            # Save previous version of f before update
            f_prev = np.copy(f)

            # Update f using Adam optimizer
            self.lr *= 1. / (1. + self.decay*i)
            lrt = self.lr * np.sqrt(1-self.beta2**i) / (1-self.beta1)
            mt = self.beta1 * mt + (1-self.beta1) * ft
            vt = self.beta2 * vt + (1-self.beta2) * ft**2
            update = lrt * mt / (np.sqrt(vt) + 1e-8)
            f -= update

            # If change in cost is below threshold, stop iterating
            if i > 7 and \
                (self.cost_[i-7]-self.cost_[i])/self.cost_[0] < self.tol:
                break

        # Print final evaluation metrics
        if self.verbose:
            print('FINAL METRICS:')
            print('-- cost = {:.4e}'.format(self.cost_[-1]))
            print('-- mse  = {:.4e}'.format(self.mse_[-1]))
            print('-- curl = {:.4e}'.format(self.curl_[-1]))

        # Set final transport map, displacements, and LOT transform
        # Note: Use previous version of f, just in case something weird
        # happened in the final iteration
        self.transport_map_ = f_prev
        self.displacements_ = f_prev - np.stack((yv,xv))
        lot = self.displacements_ * np.sqrt(sig0)

        self.is_fitted = True

        return lot


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
        transport_map : array, shape (2, height, width)
            Forward transport map.
        sig1 : 2d array, shape (height, width)
            Signal to transform.

        Returns
        -------
        sig0_recon : array, shape (height, width)
            Reconstructed reference signal sig0.
        """
        # Check inputs
        transport_map = check_array(transport_map, ndim=3,
                                    dtype=[np.float64, np.float32])
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig1, ['transport_map', 'sig1'])

        # Jacobian and its determinant
        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        # Reconstruct sig0
        sig0_recon = detJ * interp2d(sig1, transport_map, fill_value=sig1.min())

        return sig0_recon


    def apply_inverse_map(self, transport_map, sig0):
        """
        Appy inverse transport map.

        Parameters
        ----------
        transport_map : array, shape (2, height, width)
            Forward transport map. Inverse is computed in this function.
        sig0 : array, shape (height, width)
            Reference signal.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        # Check inputs
        transport_map = check_array(transport_map, ndim=3,
                                    dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig0, ['transport_map', 'sig0'])

        # Jacobian and its determinant
        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        # Let's hope there are no NaNs/Infs in sig0/detJ
        sig1_recon = griddata2d(sig0/detJ, transport_map, fill_value=sig0.min())

        return sig1_recon



class MultiVOT2D(VOT2D):
    """
    Multi-Scale Variational Optimal Transport 2D Transform.

    .. note::

    Unlike the original paper by Kundu et al., this implementation uses the
    Adam gradient descent optimizer.

    Parameters
    ----------
    n_scales : int (default=3)
        Number of scales.
    alpha : array-like or float (default=0.01)
        Regularization of the curl operator. For scalar inputs, the same value
        is used for all scales. For array-like inputs, len(alpha)=n_scales.
    lr : array-like or float (default=0.01)
        Learning rate. For scalar inputs, the same value is used for all scales.
        For array-like inputs, len(lr)=n_scales.
    beta1 : array-like or float (default=0.9)
        Adam optimizer parameter. 0 < beta1 < 1. Generally close to 1. For
        scalar inputs, the same value is used for all scales. For array-like
        inputs, len(beta1)=n_scales.
    beta2 : array-like or float (default=0.999)
        Adam optimizer parameter. 0 < beta2 < 1. Generally close to 1. For
        scalar inputs, the same value is used for all scales. For array-like
        inputs, len(beta2)=n_scales.
    decay : array-like or float (default=0.)
        Learning rate decay over each update. For scalar inputs, the same value
        is used for all scales. For array-like inputs, len(decay)=n_scales.
    max_iter : array-like or int (default=300)
        Maximum number of iterations. For scalar inputs, the same value is used
        for all scales. For array-like inputs, len(max_iter)=n_scales.
    tol : array-like or float (default=0.001)
        Stop iterating when change in cost function is below this threshold.
        For scalar inputs, the same value is used for all scales. For array-like
        inputs, len(tol)=n_scales.
    verbose : array-like or int (default=1)
        Verbosity during optimization. 0=no output, 1=print cost,
        2=print all metrics. For scalar inputs, the same value is used for all
        scales. For array-like inputs, len(verbose)=n_scales.

    Attributes
    -----------
    displacements_ : array, shape (2, height, width)
        Displacements u. First index denotes direction: displacements_[0] is
        y-displacements, and displacements_[1] is x-displacements.
    transport_map_ : array, shape (2, height, width)
        Transport map f. First index denotes direction: transport_map_[0] is
        y-map, and transport_map_[1] is x-map.
    cost_ : list of float
        Value of cost function at each iteration.
    mse_ : list of float
        Mean squared error at each iteration.
    curl_ : list of float
        Curl at each iteration.
    params_ : list of dict
        Dictionary of parameters for each scale.
    displacements_all_ : list of 3d array
        Displacements u at all scales.
    transport_map_all_ : list of 3d array
        Transport map f at all scales.
    cost_all_ : list of list
        Value of cost function at all scales.
    mse_all_ : list of list
        Mean squared error at all scales.
    curl_all_ : list of float
        Curl at all scales.

    References
    ----------
    [Discovery and visualization of structural biomarkers from MRI using
    transport-based morphometry]
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5912801/)
    [Adam - A method for stochastic optimization]
    (http://arxiv.org/abs/1412.6980v8)
    """
    def __init__(self, n_scales=3, **kwargs):
        super(MultiVOT2D, self).__init__()
        self.n_scales = n_scales

        # Create overall parameter dictionary
        for k in kwargs.keys():
            # Convert scalars to lists
            if not (isinstance(kwargs[k],list) or isinstance(kwargs[k],tuple) or
                isinstance(kwargs[k],np.ndarray)):
                kwargs[k] = [kwargs[k]] * n_scales

            # Is the list/array/tuple the same length as n_scales?
            if len(kwargs[k]) != n_scales:
                raise ValueError("Parameter {} must be a scalar or iterable of "
                                 "length n_scales={}".format(k, n_scales))

        # Split overall dictionary into separate dictionary for each scale
        self.params_ = []
        for i in range(n_scales):
            p = {}
            for k,v in kwargs.items():
                p[k] = v[i]
            self.params_.append(p)

        return


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
        lot : array, shape (2, height, width)
            LOT transform of input image sig1. First index denotes direction:
            lot[0] is y-LOT, and lot[1] is x-LOT.
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

        # Initalize attributes
        self.transport_map_all_ = []
        self.displacements_all_ = []
        self.cost_all_ = []
        self.mse_all_ = []
        self.curl_all_ = []

        # Compute the pyramid downscales
        scales = 2**np.arange(self.n_scales)[::-1]

        for i,(sc,par) in enumerate(zip(scales,self.params_)):
            # Downsample the images if scale > 1
            if sc > 1:
                sig0_dwn = pyramid_reduce(sig0, downscale=sc, cval=sig0.min())
                sig1_dwn = pyramid_reduce(sig1, downscale=sc, cval=sig1.min())
            else:
                sig0_dwn = sig0
                sig1_dwn = sig1

            # Set the initial transport map for this scale.
            # If this is the 1st scale, f_init=None (i.e. f=x), else, f_init is
            # upsampled from previous scale
            f_init = None
            if i > 0:
                f0 = resize(2*self.transport_map_all_[-1][0], sig0_dwn.shape,
                            mode='edge')
                f1 = resize(2*self.transport_map_all_[-1][1], sig0_dwn.shape,
                            mode='edge')
                f_init = np.stack((f0,f1))

            # Compute forward VOT transform for this scale
            vot = VOT2D(**par)
            lot = vot.forward(sig0_dwn, sig1_dwn, f_init=f_init)

            # Update attributes/metrics
            self.transport_map_all_.append(vot.transport_map_)
            self.displacements_all_.append(vot.displacements_)
            self.cost_all_.append(vot.cost_)
            self.mse_all_.append(vot.mse_)
            self.curl_all_.append(vot.curl_)

        # Set final attributes/metrics
        self.transport_map_ = vot.transport_map_
        self.displacements_ = vot.displacements_
        self.cost_ = vot.cost_
        self.mse_ = vot.mse_
        self.curl_ = vot.curl_

        self.is_fitted = True

        return lot
