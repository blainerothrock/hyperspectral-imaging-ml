from . import TransformException, BaseTransform

import numpy as np
from sklearn.decomposition import PCA

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

    def applyPCA(X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX, pca