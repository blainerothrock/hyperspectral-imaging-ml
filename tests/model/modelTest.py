import pytest

def test_Model(input):
    assert len(model(input)) == 16
