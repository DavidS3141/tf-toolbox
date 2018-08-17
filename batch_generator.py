import numpy as np

te = TypeError("list_datasets could not be successfully casted"
               " to valid type!")


def batch_generator(batch_size, list_datasets, shuffle=True,
                    single_epoch=False):
    '''
    Important: The list should consist of tuples, each containing numpy arrays
    with the same size. Inside each tuple the correlation stays preserved.
    '''
    if isinstance(list_datasets, np.ndarray):
        list_datasets = [(list_datasets,)]
    elif isinstance(list_datasets, tuple):
        for elem in list_datasets:
            if not isinstance(elem, np.ndarray):
                raise te
        list_datasets = [list_datasets]
    elif isinstance(list_datasets, list):
        list_datasets = list(list_datasets)
        for i in range(len(list_datasets)):
            tpl = list_datasets[i]
            if isinstance(tpl, np.ndarray):
                list_datasets[i] = (tpl,)
            elif isinstance(tpl, tuple):
                for elem in tpl:
                    if not isinstance(elem, np.ndarray):
                        raise te
            else:
                raise te
    else:
        raise te
    assert isinstance(list_datasets, list)
    for tpl in list_datasets:
        assert isinstance(tpl, tuple)
        assert isinstance(tpl[0], np.ndarray)
        n = len(tpl[0])
        for elem in tpl:
            if not len(elem) == n:
                raise ValueError("Correlated datasets need to have same size!")
            assert isinstance(elem, np.ndarray)
    nbr_datasets = len(list_datasets)
    n = len(list_datasets[0][0])
    idx_max = 0
    for i, tpl in enumerate(list_datasets):
        # measure epochs using largest dataset
        if len(tpl[0]) > n:
            idx_max = i
            n = len(tpl[0])
    batch_size = min(n, batch_size)
    if single_epoch:
        return _single_epoch_batch_generator(
            batch_size, list_datasets, shuffle, idx_max, n, nbr_datasets)
    else:
        return _infinite_epochs_batch_generator(
            batch_size, list_datasets, shuffle, idx_max, n, nbr_datasets)


def _infinite_epochs_batch_generator(
        batch_size, list_datasets, shuffle, idx_max, n, nbr_datasets):
    data_loaded = [tuple([elem[0:0] for elem in tpl]) for tpl in list_datasets]
    nbr_epochs = -1
    while True:
        for i in range(nbr_datasets):
            while len(data_loaded[i][0]) < batch_size:
                if shuffle:
                    perm = np.random.permutation(len(list_datasets[i][0]))
                else:
                    perm = range(len(list_datasets[i][0]))
                data_loaded[i] = tuple(
                    [np.concatenate((data_loaded[i][k],
                                     list_datasets[i][k][perm]))
                     for k in range(len(list_datasets[i]))])
                if i == idx_max:
                    nbr_epochs += 1
        result = []
        new_data_loaded = []
        for i in range(nbr_datasets):
            result_tpl = []
            data_loaded_tpl = []
            for k in range(len(list_datasets[i])):
                result_curr, data_loaded_curr = np.split(data_loaded[i][k],
                                                         [batch_size])
                result_tpl.append(result_curr)
                data_loaded_tpl.append(data_loaded_curr)
            result.append(tuple(result_tpl))
            new_data_loaded.append(tuple(data_loaded_tpl))
        data_loaded = new_data_loaded
        epoch_float = 1 - float(len(data_loaded[idx_max][0])) / n
        assert(epoch_float >= 0.)
        assert(epoch_float <= 1.)

        yield result, nbr_epochs + epoch_float


def _single_epoch_batch_generator(
        batch_size, list_datasets, shuffle, idx_max, n, nbr_datasets):
    if shuffle:
        shuffled_datasets = []
        for i in range(nbr_datasets):
            perm = np.random.permutation(len(list_datasets[i][0]))
            shuffled_datasets.append(tuple(
                [arr[perm, ...] for arr in list_datasets[i]]))
        list_datasets = shuffled_datasets

    nbr_iterations = n // batch_size + 1
    for k in range(nbr_iterations):
        result = []
        for i in range(nbr_datasets):
            cur_size = len(list_datasets[i][0])
            left = (k * cur_size) // nbr_iterations
            right = ((k + 1) * cur_size) // nbr_iterations
            result.append(tuple(
                [arr[left:right, ...] for arr in list_datasets[i]]))
        yield result, float(k) / nbr_iterations
