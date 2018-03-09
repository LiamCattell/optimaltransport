import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates

from .validation import check_array, assert_equal_shape


def signal_to_pdf(input, sigma=0., epsilon=1e-8, total=1.):
    """
    Get the (smoothed) probability density function of a signal.

    Performs the following operations:
    1. Smooth sigma with a Gaussian filter
    2. Normalize signal such that it sums to 1
    3. Add epsilon to ensure signal is strictly positive
    4. Re-normalize signal such that it sums to total

    Parameters
    ----------
    input : ndarray
        Input array
    sigma : scalar
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single
        number, in which case it is equal for all axes.
    epsilon : scalar
        Offset to ensure that signal is strictly positive.
    total : scalar
        Value of the signal summation.

    Returns
    -------
    pdf : ndarray
        Returned array of same shape as input
    """
    input = check_array(input, dtype=['float32','float64'])

    if sigma < 0:
        raise ValueError('sigma must be >= 0.')

    if epsilon <= 0:
        raise ValueError('epsilon must be > 0.')

    if total <= 0:
        raise ValueError('total must be > 0.')

    pdf = gaussian_filter(input, sigma)
    pdf /= pdf.sum()
    pdf += epsilon
    pdf /= pdf.sum()
    pdf *= total
    return pdf


def match_shape2d(a, b):
    """
    Crop array B such that it matches the shape of A.

    Parameters
    ----------
    a : 2d array
        Array of desired size.
    b : 2d array
        Array to crop. Shape must be larger than (or equal to) the shape
        of array a.

    Returns
    -------
    b_crop : 2d array
        Cropped version of b, with the same shape as a.
    """
    a = check_array(a, ndim=2, force_all_finite=False)
    b = check_array(b, ndim=2, force_all_finite=False)

    # Difference in dimensions
    h, w = a.shape
    ylo = (b.shape[0] - h) // 2
    xlo = (b.shape[1] - w) // 2
    if ylo < 0 or xlo < 0:
        raise ValueError("A is bigger than B: "
                         "{!s} vs {!s}".format(a.shape, b.shape))

    return b[ylo:ylo+h,xlo:xlo+w]


def interp2d(img, xi, order=2, fill_value=0.):
    # INTERP ND???
    img = check_array(img, ndim=2, force_all_finite=True)
    xi = check_array(xi, ndim=3, force_all_finite=True)

    assert_equal_shape(img, xi[0])

    h, w = img.shape
    # x = [xi[0].ravel(), xi[1].ravel()]
    out = map_coordinates(img, xi.reshape((2,h*w)), order=order, cval=fill_value)
    return out.reshape((h,w))
