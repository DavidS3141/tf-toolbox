from pkg.batch_generator import batch_generator

import numpy as np
import pytest


@pytest.fixture
def DS_1D():
    '''Returns a one-dimensional dataset with random values!'''
    return np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10])


@pytest.fixture
def DS_2D_corr():
    return (np.array([[1, -2], [3, -4], [5, -6], [7, -8], [9, -10]]),
            np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8], [-9, 10]]),)


@pytest.fixture
def DS_2D_uncorr():
    return [(np.array([[1, -2], [3, -4], [5, -6], [7, -8], [9, -10]]),),
            np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8], [-9, 10],
                      [-11, 12], [-13, 14]])]


# TODO implement complex test
@pytest.fixture
def DS_2D_complex():
    return [(np.array([[1, -2], [3, -4], [5, -6], [7, -8], [9, -10]]),
             np.array([[-1, 2, 0], [-3, 4, 0], [-5, 6, 0], [-7, 8, 0],
                       [-9, 10, 0]]),),
            (np.array([[-1, 2], [-3, 4], [-5, 6], [-7, 8], [-9, 10],
                       [-11, 12], [-13, 14]]),)]


def test_batch_generator_simple(DS_1D):
    bs = 3
    bg = batch_generator(bs, DS_1D)
    all_concat = np.empty((0,), dtype=DS_1D.dtype)
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


def test_batch_generator_corr(DS_2D_corr):
    bs = 3
    bg = batch_generator(bs, DS_2D_corr)
    all_concat_1 = np.empty((0, 2), dtype=DS_2D_corr[0].dtype)
    all_concat_2 = np.empty((0, 2), dtype=DS_2D_corr[1].dtype)
    for i in range(5):
        n = next(bg)
        assert isinstance(n, tuple)
        assert len(n) == 2
        batch, epoch = n
        assert isinstance(epoch, float)
        assert pytest.approx(epoch) == (i + 1) * 3. / 5.
        assert isinstance(batch, list)
        assert len(batch) == 1
        assert isinstance(batch[0], tuple)
        assert len(batch[0]) == 2
        assert isinstance(batch[0][0], np.ndarray)
        assert isinstance(batch[0][1], np.ndarray)
        assert batch[0][0].shape == (3, 2)
        assert batch[0][1].shape == (3, 2)
        assert np.all(batch[0][0] == -batch[0][1])
        all_concat_1 = np.concatenate([all_concat_1, batch[0][0]])
        all_concat_2 = np.concatenate([all_concat_2, batch[0][1]])
    assert all_concat_1.shape == (15, 2)
    assert all_concat_2.shape == (15, 2)
    for i in range(3):
        data1 = all_concat_1[i*5:i*5+5, :]
        data2 = all_concat_2[i*5:i*5+5, :]
        assert np.all(data1[np.argsort(data1[:, 0]), :] == DS_2D_corr[0])
        assert np.all(data2[np.argsort(-data2[:, 0]), :] == DS_2D_corr[1])


def test_batch_generator_uncorr(DS_2D_uncorr):
    bs = 3
    bg = batch_generator(bs, DS_2D_uncorr)
    all_concat_1 = np.empty((0, 2), dtype=DS_2D_uncorr[0][0].dtype)
    all_concat_2 = np.empty((0, 2), dtype=DS_2D_uncorr[1].dtype)
    nbr_correlated = 0
    for i in range(35):
        n = next(bg)
        assert isinstance(n, tuple)
        assert len(n) == 2
        batch, epoch = n
        assert isinstance(epoch, float)
        assert pytest.approx(epoch) == (i + 1) * 3. / 7.
        assert isinstance(batch, list)
        assert len(batch) == 2
        assert isinstance(batch[0], tuple)
        assert isinstance(batch[1], tuple)
        assert len(batch[0]) == 1
        assert len(batch[1]) == 1
        assert isinstance(batch[0][0], np.ndarray)
        assert isinstance(batch[1][0], np.ndarray)
        assert batch[0][0].shape == (3, 2)
        assert batch[1][0].shape == (3, 2)
        nbr_correlated += np.all(batch[0][0] == -batch[1][0])
        all_concat_1 = np.concatenate([all_concat_1, batch[0][0]])
        all_concat_2 = np.concatenate([all_concat_2, batch[1][0]])
    assert nbr_correlated <= 2
    assert all_concat_1.shape == (105, 2)
    assert all_concat_2.shape == (105, 2)
    for i in range(3):
        data1 = all_concat_1[i*5:i*5+5, :]
        data2 = all_concat_2[i*7:i*7+7, :]
        assert np.all(data1[np.argsort(data1[:, 0]), :] == DS_2D_uncorr[0][0])
        assert np.all(data2[np.argsort(-data2[:, 0]), :] == DS_2D_uncorr[1])


def test_batch_generator_type_error_1():
    with pytest.raises(TypeError):
        next(batch_generator(3, ("3", 2)))


def test_batch_generator_type_error_2():
    with pytest.raises(TypeError):
        next(batch_generator(3, [("3", 2), ("3", 2)]))


def test_batch_generator_type_error_3():
    with pytest.raises(TypeError):
        next(batch_generator(3, ["3"]))


def test_batch_generator_type_error_4():
    with pytest.raises(TypeError):
        next(batch_generator(3, 3))


def test_batch_generator_value_error():
    with pytest.raises(ValueError):
        next(batch_generator(3, (np.empty((3, 5)), np.empty((5, 3)))))
