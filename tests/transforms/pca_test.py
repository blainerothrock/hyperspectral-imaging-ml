import pytest
from hyperspec.transforms import PCA, BaseTransform, TransformException
import numpy as np

class TestPCATransform:

    # TODO: placeholder tests
    def test_pca(self):
        oldX = np.zeros((255,255,255))
        num_components = 2
        tf = PCA(num_components=num_components)
        data = {'raw': (oldX, None)}
        tf(data)
        newX, _ = data['raw']
        assert newX is not None
        assert oldX.shape != newX.shape
        assert newX.shape[2] == num_components
