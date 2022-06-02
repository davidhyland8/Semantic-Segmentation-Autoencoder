import torch as T
import numpy as np
from torch.utils.data import Dataset

class MaskDataset(Dataset):
    def __init__(self, mask_paths, size, num_classes, device):
        self.mask_paths = mask_paths
        self.size = size
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask = np.load(self.mask_paths[idx])
        im = T.from_numpy(mask).float().to(self.device)
        im = T.reshape(im, (1, 128, 128))
        mask = T.from_numpy(mask).long().to(self.device)
        mask = T.reshape(mask, (128, 128))
        return im, mask