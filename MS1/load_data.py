import sys, os
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from data.musdb_loader import get_musdb_folds
from data.dataset import RawSeparationDataset

def load_data(dataset='musdb', transformation=None, n_train=None, n_test=None, datasets_path='/home/space/datasets'):
    '''
    :param dataset: Name of the dataset to load (default: musdb)
    :transformation: callable object that is applied to every train and test sample
    :n_train: Number of train samples to load (by default loads all train samples).
    :n_test: Number of train samples to load (by default loads all test samples).
    :datasets_path: Root directory where datasets are located.
    :returns: A generator pair. The first generator returns pairs (x,y) from the training set,
    second generator returns pairs from the test set,
    '''
    db_path = os.path.join(datasets_path, dataset)
    if (not os.path.exists(db_path)):
        raise ValueError(f"Dataset {dataset} not found in {db_path}")
    transformation = [] if transformation is None else [transformation]
    train_dataset = RawSeparationDataset(get_musdb_folds(db_path)['train'][:n_train],
                                         transforms=transformation)
    test_dataset = RawSeparationDataset(get_musdb_folds(db_path)['test'][:n_test],
                                        transforms=transformation)

    return train_dataset, test_dataset