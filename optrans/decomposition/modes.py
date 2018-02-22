import numpy as np

from ..utils import check_array, assert_equal_shape, check_decomposition


def get_mode_variation(decomp, component=0, n_std=3., n_steps=5):
    """
    Get the mode of variation along a given direction through the data.

    Parameters
    ----------
    decomp : object
        A trained PCA or PLDA object.
    component : int (default=0.)
        Index of the desired component.
    n_std : float (default=3.)
        Range of standard deviations along the direction. For example if
        n_std=3, the returned data will correspond to the range -1.5*std to
        +1.5*std
    n_steps : int (default=5)
        Number of steps along the direction.

    Returns
    -------
    mode : array, shape (n_steps, n_features)
        Reconstructed data along desired direction.
    """
    # Check that decomp is a PCA or PLDA object
    mean, comp, std = check_decomposition(decomp)
    comp = comp[component]
    std = std[component]

    # Initialise the mode array and the steps along the direction
    mode = np.zeros((n_steps,mean.size))
    alpha = np.linspace(-0.5*n_std*std, 0.5*n_std*std, n_steps)

    # Reconstruct data along direction
    for i,a in enumerate(alpha):
        mode[i] = mean + a * comp

    return mode


def get_mode_histogram(X, y=None, component=0, n_bins=10, rng=None):
    """
    Compute the normalized histogram of the data projected on to a given
    component.

    Parameters
    ----------
    X : array, shape (n_samples, n_components)
        Data.
    y : array, shape (n_samples,) or None
        Class labels. If class labels are provided, this function will compute
        a separate histogram for each class.
    component : int (default=0)
        Index of the desired component.
    n_bins : int (default=10)
        Number of histogram bins.
    rng : (float, float) or None
        The lower and upper range of the bins. If not provided, rng is simply
        (X.min(), X.max()). The first element of the range must be less than or
        equal to the second.

    Returns
    -------
    hist : array, shape (n_bins,) or list of arrays, length (n_classes)
        The normalized values of the histogram. If class labels are provided,
        hist is a list of arrays (one for each class).
    bin_centers : array, shape (n_bins,)
        Histogram bin centers. The bin centers are the same for all classes.
    """
    X = check_array(X, ndim=2)

    # Default to "all data from class 0"
    if y is None:
        y = np.zeros(X.shape[0])

    # Check array dimensions match
    y = check_array(y, ndim=1)
    if y.size != X.shape[0]:
        raise ValueError("Number of samples in X and y does not match.")

    # Class labels
    labels = np.unique(y)

    # Initialize histogram and its range (must have same range for each class)
    hist = []
    if rng is None:
        rng = (X[:,component].min(), X[:,component].max())

    # Get the histogram for each class
    for lab in labels:
        h, bin_edges = np.histogram(X[y==lab,component], bins=n_bins, range=rng)
        hist.append(h/h.sum())
        bin_centers = bin_edges[1:] - np.diff(bin_edges)/2.

    if labels.size == 1:
        return hist[0], bin_centers
    else:
        return hist, bin_centers
