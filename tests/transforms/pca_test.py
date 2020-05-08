import pytest
from hyperspec.transforms import PCA, BaseTransform, TransformException
import numpy as np

class TestPCATransform:

    # TODO: placeholder tests
    def test_pca(self):
        oldX = np.zeros((255,255,255))
        numComponents = 75
        newX, pca = PCA.apply_pca(oldX, numComponents)
        assert newX is not None
        assert pca is not None
        assert oldX.shape != newX.shape
        assert newX.shape[2] == numComponents
