from pkg.nets.tf_activations import get_activation

import pytest


precomputed_activation_results = {
    'elu': {(None,): (1.3219389963, 0.2504223680)},
    'leaky_relu': {(0.01,): (1.7048316272, 0.6733281229),
                   (0.2,): (1.5464600604, 0.4935586424)},
    'relu': {(None,): (1.7128585504, 0.6833316961)},
    'relu6': {(None,): (1.7138376616, 0.6836224424)},
    'selu': {(None,): (1.0000000000, -0.0000000000)},
    'sigmoid': {(None,): (4.0000000001, 0.5000000000)},
    'sigmoid_num': {(None,): (1.3126771368, 2.0000000000)},
    'softplus': {(None,): (1.8438395187, 1.0211922094)},
    'softsign': {(None,): (1.0000100000, 0.0000000000)},
    'softsign_num': {(None,): (1.4229583818, 0.0000000000)},
    'tanh': {(None,): (1.0000000000, 0.0000000000)},
    'tanh_num': {(None,): (0.6563385684, 0.0000000000)},
    'identity': {(None,): (1.0000000000, 0.0000000000)},
}


def test_activations():
    get_activation._hashed_activations = dict()
    for act_name in precomputed_activation_results:
        for params in precomputed_activation_results[act_name]:
            true_pre_std, true_post_mean = \
                precomputed_activation_results[act_name][params]
            act, pre_std, post_mean = get_activation(act_name, params)
            assert pytest.approx(pre_std, abs=1e-6) == true_pre_std
            assert pytest.approx(post_mean, abs=1e-6) == true_post_mean


def test_hashed_activations():
    get_activation._hashed_activations = dict()
    for act_name in precomputed_activation_results:
        for params in precomputed_activation_results[act_name]:
            true_pre_std, true_post_mean = \
                precomputed_activation_results[act_name][params]
            act, pre_std, post_mean = get_activation(act_name, params)
            assert pytest.approx(pre_std, abs=1e-6) == true_pre_std
            assert pytest.approx(post_mean, abs=1e-6) == true_post_mean
