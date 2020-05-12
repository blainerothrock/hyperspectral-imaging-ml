import pytest
from hyperspec.transforms import ImageTransform, BaseTransform, TransformException
import numpy as np

class TestImageTransform:

    def test_padWithZeros(self):
        nonPadX = np.zeros((5,5,5))
        margin = 2
        IT = ImageTransform()
        padX = IT._pad_with_zeros(nonPadX, margin)
        assert nonPadX.shape != padX.shape
        assert nonPadX.shape == (5,5,5)
        assert padX.shape == (nonPadX.shape[0]+ 2*margin, nonPadX.shape[1]+ 2*margin, 5)
        pass

    def test_createImageCubes(self):
        X = np.zeros((50, 50, 10))
        Y = np.ones((50, 50))
        window_size = 2
        IT = ImageTransform(window_size=window_size, inplace=True)
        data = {'raw': (X, Y)}
        data = IT(data)
        patchesX, patchesY = data['raw']
        assert X.shape != patchesX.shape
        assert Y.shape != patchesY.shape
        assert patchesX.shape[0] == X.shape[0]**2
        assert patchesX.shape[1] == patchesX.shape[2] == window_size
        assert patchesX.shape[3] == X.shape[2]
        assert patchesY.shape[0] == patchesX.shape[0]
        pass
