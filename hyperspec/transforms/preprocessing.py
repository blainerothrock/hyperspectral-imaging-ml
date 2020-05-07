from . import TransformException, BaseTransform

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class Preprocessing(BaseTransform):
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

    # def splitTrainTestSet(X, y, testRatio, randomState=345):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    #     return X_train, X_test, y_train, y_test

    def applyPCA(X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
        return newX, pca

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
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