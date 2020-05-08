import pytest
from hyperspec.transforms import to_tensor, TransformException
import torch
import numpy as np


class TestToTensor:

    def test_to_tensor(self, signal):
        device = torch.device('cpu')
        tf = to_tensor(device=device)
        data = {tf.source: signal}
        tf(data)
        out = data[tf.output]

        assert isinstance(out, torch.Tensor), 'should be tensor'
        assert np.allclose(signal.data, out.numpy()), 'tensor should be equal to original signal'

    def test_params(self):
        with pytest.raises(TransformException):
            tf = to_tensor(device=0)

    def test_print(self):
        string = str(to_tensor(device=torch.device('cpu')))
        assert 'device' in string