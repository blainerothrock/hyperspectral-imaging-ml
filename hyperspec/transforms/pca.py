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
            num_components=75,
            source='raw',
            output='output',
            inplace=True):

        super().__init__(source, output, inplace)
        self.num_components = num_components

    def __call__(self, data):
        super().__call__(data)

        X, y = data[self.source]
        X = self._apply_pca(X)

        super().update(data, (X, y))


    def _apply_pca(self, X):
        """
        Applies PCA to the input array X.
        :param X: A 3-dimensional numpy array of size (x, y, z) where x == y == z.
        :param numComponents: The number of components to keep.
        :return: A 3-dimensional numpy array of size (x, y, numComponents) and the PCA object itself.
        """
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = sklearnPCA(n_components=self.num_components, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], self.num_components))
        return newX