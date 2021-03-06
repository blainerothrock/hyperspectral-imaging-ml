from . import TransformException, BaseTransform

import os, shutil
import numpy as np
import torch



class ImageTransform(BaseTransform):
    """
    Preprocesses the raw images.
    Args:
        X: (numpy array): A 3-dimensional numpy array of size (x, y, z) where x == y and z == numComponents.
        y (numpy array): A 2-dimensional numpy array of size (x, y) where x == y == X.shape[0] == X.shape[1].
        window_size (int): The size of the patches to extract features from the original feature input.
        removeZeroLabels (Bool): Boolean which determines whether or not zero labels are removed.
        source (string): the data source to filter, default: raw
        output (string): optional, the key of the output data in the dictionary, default: filtered
        inplace (Bool): optional, will overwrite the source data with the trim
    Raises:
        TransformException: if a filter in not in a supporting list
    Returns:
        patchesData (numpy array): A 4-dimensional numpy array of size (x, y, z, w)
                    where x == X.shape[0]**2 and y == z == windowSize and w == numComponents.
        patchesLabel (numpy array): A 1-dimensional numpy array of size (x)
                    where x == patchesData.shape[0] == X.shape[0]**2.
    """

    def __init__(self,
                 window_size=5,
                 removeZeroLabels=True,
                 cache_dir=os.path.expanduser('~/.hyperspec/tmp/'),
                 source='raw',
                 output='transformed',
                 inplace=False):

        super().__init__(source, output, inplace)

        self.windowSize = window_size
        self.removeZeroLabels = removeZeroLabels
        self.cache_dir = cache_dir

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)

        os.makedirs(cache_dir)

    def __call__(self, data):

        # check if data contains 'labels'
        # {'src': (img, label)}

        super().__call__(data)

        img, labels = data[self.source]
        img = img.copy()
        labels = labels.copy()
        img, labels = self._create_image_cubes(img, labels)

        return super().update(data, (img, labels))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'X: {self.X}, '
            f'y: {self.y}, '
            f'windowSize: {self.windowSize}, '
            f'removeZeroLabels: {self.removeZeroLabels})'
        )

    def _pad_with_zeros(self, X, margin):
        """
        Pads a 3-dimensional numpy array with zeros around the first two dimensions.
        :param X: A 3-dimensional numpy array of size (x, y, z) where x == y == z.
        :param margin: The amount of padding applied to the first two dimensions -- third dimension left unchanged.
        :return: A padded numpy array of shape (x+margin, y+margin, z)
        """
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def _create_image_cubes(self, X, y):
        """
        :param X: (features) A 3-dimensional numpy array of size (x, y, z) where x == y and z == numComponents.
        :param y: (labels) A 2-dimensional numpy array of size (x, y) where x == y == X.shape[0] == X.shape[1].
        :param windowSize: The size of the patches to extract features from the original feature input.
        :param removeZeroLabels: Boolean which determines whether or not zero labels are removed.
        :return: patchesData: (feature patches) A 4-dimensional numpy array of size (x, y, z, w)
                    where x == X.shape[0]**2 and y == z == windowSize and w == numComponents.
                 patchesLabels: (label patches) A 1-dimensional numpy array of size (x)
                    where x == patchesData.shape[0] == X.shape[0]**2.
        """
        margin = int((self.windowSize - 1) / 2)
        zeroPaddedX = self._pad_with_zeros(X, margin=margin)
        # split patches
        patchesData = np.zeros((X.shape[0] * X.shape[1], self.windowSize, self.windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchIndex = patchIndex + 1

        if self.removeZeroLabels:
            patchesData = patchesData[patchesLabels > 0, :, :, :]
            patchesLabels = patchesLabels[patchesLabels > 0]
            patchesLabels -= 1
        return patchesData, patchesLabels

