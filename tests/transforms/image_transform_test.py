import pytest
from hyperspec.transforms import ImageTransform, BaseTransform, TransformException
import numpy as np

class TestImageTransform:

    # TODO: placeholder tests
    def test_padWithZeros(self):
        nonPadX = np.zeros((5,5,5))
        padX = ImageTransform.padWithZeros(nonPadX)
        assert nonPadX.shape != padX.shape
        assert nonPadX.shape == (5,5,5)
        assert padX.shape == (9,9,5)
        pass
