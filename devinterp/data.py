import numpy as np


class CustomDataLoader:
    """
    Dataloader which idempotently shuffles the data indices on each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        
        if shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.indices = np.arange(len(self.dataset))
        np.random.shuffle(self.indices)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            yield [self.dataset[idx] for idx in batch_indices]
