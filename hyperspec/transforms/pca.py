from . import TransformException, BaseTransform

import numpy as np
from sklearn.decomposition import PCA as sklearnPCA

class PCA(BaseTransform):
    """
    Args:
    Raises:
        TransformException
    Returns:
    """

    def __init__(
            self,
    source,
    output,
    inplace):
        super().__init__(source, output, inplace)

    def apply_pca(X, numComponents=75):
        """
        Applies PCA to the input array X.
        :param X: A 3-dimensional numpy array of size (x, y, z) where x == y == z.
        :param numComponents: The number of components to keep.
        :return: A 3-dimensional numpy array of size (x, y, numComponents) and the PCA object itself.
        """
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = sklearnPCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX, pca