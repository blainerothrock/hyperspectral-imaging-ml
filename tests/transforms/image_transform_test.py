import pytest
from hyperspec.transforms import ImageTransform, BaseTransform, TransformException
import numpy as np

class TestImageTransform:

    def test_padWithZeros(self):
        nonPadX = np.zeros((5,5,5))
        margin = 2
        padX = ImageTransform.pad_with_zeros(nonPadX, margin)
        assert nonPadX.shape != padX.shape
        assert nonPadX.shape == (5,5,5)
        assert padX.shape == (nonPadX.shape[0]+ 2*margin, nonPadX.shape[1]+ 2*margin, 5)
        pass

    # TODO: placeholder tests
    def test_createImageCubes(self):
        assert True
        pass
