import numpy as np


def batch_generator(batch_size, list_datasets):
    '''
    Important: all datasets in the list should have the same size, as they
    become randomly sampled preserving the correlation between the datasets,
    meaning that data entries in the same position of two sets keep occuring
    together in the same batch at the same position.
    '''
    if not isinstance(list_datasets, list):
        list_datasets = [list_datasets]
    nbr_datasets = len(list_datasets)
    n = len(list_datasets[0])
    for data in list_datasets:
        assert(n == len(data))
    perm = np.random.permutation(n)
    data_loaded = [data[perm] for data in list_datasets]
    nbr_epochs = 0
    while True:
        while len(data_loaded[0]) < batch_size:
            perm = np.random.permutation(n)
            for i in range(nbr_datasets):
                data_loaded[i] = np.concatenate((
                    data_loaded[i], list_datasets[i][perm]))
            nbr_epochs += 1
        result = []
        new_data_loaded = []
        for i in range(nbr_datasets):
            result_curr, data_loaded_curr = np.split(data_loaded[i],
                                                     [batch_size])
            result.append(result_curr)
            new_data_loaded.append(data_loaded_curr)
        data_loaded = new_data_loaded
        epoch_float = 1 - float(len(data_loaded[0])) / n
        assert(epoch_float >= 0.)
        assert(epoch_float <= 1.)

        yield tuple(result + [nbr_epochs + epoch_float])
