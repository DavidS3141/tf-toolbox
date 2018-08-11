from batch_generator import batch_generator

import numpy as np
import pytest


@pytest.fixture
def DS_1D():
    '''Returns a one-dimensional dataset with random values!'''
    return [(np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10]),)]


@pytest.fixture
def DS_2D_corr():
    return [(np.array([[1, -2], [3, -4], [5, -6], [7, -8], [9, -10]]),
             np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8], [-9, 10]]),)]


@pytest.fixture
def DS_2D_uncorr():
    return [(np.array([[1, -2], [3, -4], [5, -6], [7, -8], [9, -10]]),),
            (np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8], [-9, 10]]),)]


@pytest.fixture
def DS_2D_complex():
    return [(np.array([[1, -2], [3, -4], [5, -6], [7, -8], [9, -10]]),
             np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8], [-9, 10]]),),
            (np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8], [-9, 10]]),)]


def test_batch_generator_simple(DS_1D):
    bs = 3
    bg = batch_generator(bs, DS_1D)
    all_concat = np.empty((0,))
    for i in range(10):
        batch, epoch = next(bg)
        assert isinstance(epoch, float)
        assert pytest.approx(epoch) == (i + 1) * 3. / 10.
        assert isinstance(batch, list)
        assert len(batch) == 1
        assert isinstance(batch[0], tuple)
        assert len(batch[0]) == 1
        assert isinstance(batch[0][0], np.ndarray)
        assert batch[0][0].shape == (3,)
        all_concat = np.concatenate([all_concat, batch[0][0]])
    assert all_concat.shape == (30,)
    for i in range(3):
        data = all_concat[i*10:i*10+10]
        assert np.all(np.sort(data) == np.sort(DS_1D))
