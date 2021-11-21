import os
from torch.utils.data import Dataset
from tqdm import tqdm
from waveUNet.data.utils import load
import logging
logger = logging.getLogger(__name__)

class SeparationDataset(Dataset):
    def __init__(self, data_path_list, domain='time', transforms=[]):
        '''
        Initialises a source separation dataset
        :param data_path_list: list which contains data_mix path and data_accompaniment paths
        :param frame_shapes: todo, deal with output frames
        :param transforms: list of generator functions
        '''
        super(SeparationDataset, self).__init__()

        self.data_path_list = data_path_list
        self.length = len(data_path_list)
        self.transforms = transforms
        logger.info(f"Intialized dataset of size {self.length}. The following transformation functions will be applied. {self.transforms}")

    def __getitem__(self, index):
        paths = self.data_path_list[index]
        mixed_path = paths['mix']
        #load returns tensor and sample rate as a tuple
        mixed = load(mixed_path, mode="pytorch")[0]
        targets = {source: load(path, mode="pytorch")[0] for source, path in paths.items() if source != "mix"}

        if len(self.transforms) > 0:
            for transform in self.transforms:
                mixed, targets = transform(mixed, targets)
        return mixed, targets

    def __len__(self):
        return self.length
