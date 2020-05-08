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

    def padWithZeros(self, margin=2):
        newX = np.zeros((self.shape[0] + 2 * margin, self.shape[1] + 2 * margin, self.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:self.shape[0] + x_offset, y_offset:self.shape[1] + y_offset, :] = self
        return newX

    def createImageCubes(self, y, windowSize=5, removeZeroLabels = True):
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self.padWithZeros(self, margin=margin)
        # split patches
        patchesData = np.zeros((self.shape[0] * self.shape[1], windowSize, windowSize, self.shape[2]))
        patchesLabels = np.zeros((self.shape[0] * self.shape[1]))
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