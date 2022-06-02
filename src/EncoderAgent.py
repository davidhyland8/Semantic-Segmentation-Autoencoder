import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from random import sample

from SegmentationDataset import SegmentationDataset
from MaskDataset import MaskDataset
from encoder import autoencoder
import torch as T
from torch.nn import CrossEntropyLoss
from TverskyCrossEntropyDiceWeightedLoss import TverskyCrossEntropyDiceWeightedLoss


def load_data(path, masktomask):
    """
    Helper method to load the dataset
    :param path: Path to location of dataset
    :return: lists of all the images and masks
    """
    dirs = ['Clear Noon Dry', 'Cloudy Evening HR', 'Cloudy Evening LR', 'Night Dry', 'Night HR', 'Night LR']
    all_images = []
    all_masks = []

    for d in dirs:
        images_list = list(path.glob(f'{d}/RGB/*.npy'))
        images_list.sort()
        masks_list = list(path.glob(f'{d}/Semantic/*.npy'))
        masks_list.sort()
        if len(images_list) != len(masks_list):
            raise ValueError('Invalid data loaded')
        pairs = zip(images_list, masks_list)
        pairs = sample(list(pairs), 5000)
        images_list, masks_list = zip(*pairs)
        all_images += images_list
        all_masks += masks_list
    
    all_images = np.array(all_images)
    all_masks = np.array(all_masks)

    if masktomask:
        return all_masks, all_masks
    else:
        return all_images, all_masks


class EncoderAgent:
    def __init__(self, val_percentage, test_num, channels, num_classes,
                 batch_size, img_size, data_path, shuffle_data,
                 learning_rate, device, tv, masktomask=False):
        """
        A helper class to facilitate the training of the model
        """
        self.device = device
        self.channels = channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.masktomask = masktomask
        self.images_list, self.masks_list = load_data(data_path, self.masktomask)
        train_split, val_split, test_split = self.make_splits(
                val_percentage, test_num, shuffle_data)
        self.train_loader = self.get_dataloader(train_split)
        self.validation_loader = self.get_dataloader(val_split)
        self.test_loader = self.get_dataloader(test_split)
        self.weight = T.tensor([0.0833, 0.0762, 0.0833, 0.0665, 0.0833, 0.0820, 0.0816, 0.0332, 0.0819, 0.0817, 0.0833, 0.0803, 0.0833]).float().to(self.device)
        self.model = autoencoder(self.channels, self.num_classes) #, learning_rate, 'rgb-sem-runs', 'rgb-sem-models', T.tensor([0.0833, 0.0762, 0.0833, 0.0665, 0.0833, 0.0820, 0.0816, 0.0332, 0.0819, 0.0817, 0.0833, 0.0803, 0.0833], dtype=T.float64))
        if tv:
            self.criterion = TverskyCrossEntropyDiceWeightedLoss(self.num_classes,
                                                             self.device)
        else:
            self.criterion = CrossEntropyLoss(self.weight)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def make_splits(self, val_percentage=0.2, test_num=10, shuffle=True):
        """
        Split the data into train, validation and test datasets
        :param val_percentage: A decimal number which tells the percentage of
                data to use for validation
        :param test_num: The number of images to use for testing
        :param shuffle: Shuffle the data before making splits
        :return: tuples of splits
        """
        if shuffle:
            shuffle_idx = np.random.permutation(range(len(self.images_list)))
            self.images_list = self.images_list[shuffle_idx]
            self.masks_list = self.masks_list[shuffle_idx]
            # np.random.shuffle(self.images_list)
            # np.random.shuffle(self.masks_list)

        val_num = len(self.images_list) - int(
            val_percentage * len(self.images_list))

        train_images = self.images_list[:val_num]
        train_masks = self.masks_list[:val_num]

        validation_images = self.images_list[val_num:-test_num]
        validation_masks = self.masks_list[val_num:-test_num]

        test_images = self.images_list[-test_num:]
        test_masks = self.masks_list[-test_num:]

        return (train_images, train_masks), \
               (validation_images, validation_masks), \
               (test_images, test_masks)

    def get_dataloader(self, split):
        """
        Create a DataLoader for the given split
        :param split: train split, validation split or test split of the data
        :return: DataLoader
        """
        if not self.masktomask:
            return DataLoader(SegmentationDataset(split[0], split[1], self.img_size,
                                              self.num_classes, self.device),
                          self.batch_size, shuffle=True)
        else:
            return DataLoader(MaskDataset(split[0], self.img_size,
                                              self.num_classes, self.device),
                          self.batch_size, shuffle=True)