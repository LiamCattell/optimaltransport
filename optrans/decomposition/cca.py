import numpy as np
import sklearn.cross_decomposition.CCA as CanonCorr

from optrans.utils import check_array


class CCA():
    def __init__(self, n_components=1, scale=True, max_iter=500, tol=1e-6,
                 copy=True):
        self.is_fitted = False
        self.n_components_ = n_components
        self.cca = CanonCorr(n_components=n_components, scale=scale,
                             max_iter=max_iter, tol=tol, copy=copy)
        return


    def _check_is_fitted(self):
        if not self.is_fitted:
            raise AssertionError("The fit function has not been "
                                 "called yet. Call 'fit' before using "
                                 "this method".format(type(self).__name__))
        return


    def fit(self, X, Y):
        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)
        Y = check_array(Y, ndim=2, dtype='numeric', force_all_finite=True)

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must be the same: "
                             "{} vs {}".format(X.shape[0], Y.shape[0]))

        self.cca.fit(X, Y)
        self.is_fitted = True
        return


    def transform(self, X, Y=None, copy=True):
        self._check_is_fitted()

        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)
        if Y is not None:
            Y = check_array(Y, ndim=2, dtype='numeric', force_all_finite=True)

        return self.cca.transform(X, Y=Y, copy=copy)


    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X, Y=Y)


    def score(self, X, Y):
        x_trans, y_trans = self.transform(X, Y)

        score = np.zeros(self.n_components_)
        for i in range(self.n_components_):
            score[i] = np.corrcoef(x_scores[:,i], y_scores[;,i])[0,1]

        if self.n_components_ == 1:
            return score[i]
        else:
            return score
