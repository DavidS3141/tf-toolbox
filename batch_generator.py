import numpy as np


def batch_generator(batch_size, list_datasets):
    '''
    Important: The list should consist of tuples, each containing numpy arrays
    with the same size. Inside each tuple the correlation stays preserved.
    '''
    if isinstance(list_datasets, np.ndarray):
        list_datasets = [(list_datasets,)]
    elif isinstance(list_datasets, tuple):
        for elem in list_datasets:
            assert isinstance(elem, np.ndarray)
        list_datasets = [list_datasets]
    elif isinstance(list_datasets, list):
        for i in range(len(list_datasets)):
            tpl = list_datasets[i]
            if isinstance(tpl, np.ndarray):
                list_datasets[i] = (tpl,)
            elif isinstance(tpl, tuple):
                for elem in tpl:
                    assert isinstance(elem, np.ndarray)
            else:
                raise TypeError("list_datasets could not be successfully casted"
                                " to valid type!")
    else:
        raise TypeError("list_datasets could not be successfully casted to "
                        "valid type!")
    assert isinstance(list_datasets, list)
    for tpl in list_datasets:
        assert isinstance(tpl, tuple)
        assert isinstance(tpl[0], np.ndarray)
        n = len(tpl[0])
        for elem in tpl:
            assert len(elem) == n
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
    data_loaded = [tuple([elem[0:0] for elem in tpl]) for tpl in list_datasets]
    nbr_epochs = -1
    while True:
        for i in range(nbr_datasets):
            while len(data_loaded[i][0]) < batch_size:
                perm = np.random.permutation(len(list_datasets[i][0]))
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
