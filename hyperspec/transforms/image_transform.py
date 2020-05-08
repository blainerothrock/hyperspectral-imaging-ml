from . import TransformException, BaseTransform

import numpy as np

class ImageTransform(BaseTransform):
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


    def pad_with_zeros(X, margin=2):
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

    def create_image_cubes(X, y, windowSize=5, removeZeroLabels = True):
        """
        :param X: A 3-dimensional numpy array of size (x, y, z) where x == y == z.
        :param y:
        :param windowSize:
        :param removeZeroLabels:
        :return:
        """
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = X.padWithZeros(X, margin=margin)
        # split patches
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1
        return patchesData, patchesLabels