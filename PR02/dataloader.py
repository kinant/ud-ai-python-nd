# PROGRAMMER: Kinan Turman
# DATE UPDATED: Oct. 22, 2025
# PURPOSE: Defining a class for loading image datasets

import torch
from torchvision import datasets, transforms, models

import json
from os import path

import helpers.utils as utils

# IMAGE CONSTANTS
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
IMG_RESIZE = (255, 255)
BATCH_SIZE = 64
IMG_ROTATION = 45

# DATASET CONSTANTS
TRAIN = "train"
TEST = "test"
VALID = "valid"

DATASETS = {
    TRAIN: TRAIN,
    VALID: VALID,
    TEST: TEST,
}

class ImageDataLoader:

    @property
    def data_transforms(self):
        return self._data_transforms

    @property
    def image_datasets(self):
        return self._image_datasets

    @property
    def dataloaders(self):
        return self._dataloaders

    @property
    def dataset_sizes(self):
        return self._dataset_sizes

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def cat_to_name(self):
        return self._cat_to_name

    def __init__(self, datadir, batch_size=BATCH_SIZE, load_data=True):
        """
        Init function for class
        :param datadir: directory of data to load
        :param batch_size: batch size to use
        :param load_data: flag to load data or not
        """

        # define and init class attributes
        self._datadir = datadir
        self._batch_size = batch_size

        self._data_transforms = {}

        self._data_dirs = {}
        self._image_datasets = {}
        self._dataset_sizes = {}
        self._dataloaders = {}

        self._class_names = []
        self._num_classes = 0
        self._cat_to_name = {}

        # proceed to load data
        if load_data:
            self._load_data()

    def _build_data_transforms(self):
        """
        Build data transforms
        """
        print(f"Building data transforms...")

        # define separate data transforms for training, validation and test data
        self._data_transforms = {
            DATASETS[TRAIN]: transforms.Compose([transforms.RandomRotation(IMG_ROTATION),
                                                   transforms.RandomResizedCrop(IMG_SIZE),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(utils.NORMALIZING_MEAN,
                                                                        utils.NORMALIZING_STD)]),

            DATASETS[VALID]: transforms.Compose([transforms.Resize(IMG_RESIZE),
                                                   transforms.CenterCrop(IMG_SIZE),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(utils.NORMALIZING_MEAN,
                                                                        utils.NORMALIZING_STD)]),

            DATASETS[TEST]: transforms.Compose([transforms.Resize(IMG_RESIZE),
                                                  transforms.CenterCrop(IMG_SIZE),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(utils.NORMALIZING_MEAN, utils.NORMALIZING_STD)])
        }

    def _load_data(self):
        """
        Load data
        """
        print(f"Loading data...")

        # define the data dirs, as in "flowers/train", "flowers/test", "flowers/valid"
        self._data_dirs = {
            key: path.join(self._datadir, DATASETS[key])
            for key in DATASETS.keys()
        }

        # build the data transforms
        self._build_data_transforms()

        print(f"Data Dirs: {self._data_dirs}")

        # define the image datasets using datasets>ImageFolder
        print(f"Loading image datasets...")
        self._image_datasets = {
            key: datasets.ImageFolder(self._data_dirs[key], transform=self.data_transforms[key])
            for key in DATASETS.keys()
        }

        # determine dataset sizes
        self._dataset_sizes = {key: len(self._image_datasets[key]) for key in DATASETS.keys()}

        print(f"Creating data loaders...")
        # create the dataloaders
        self._dataloaders = {

            DATASETS[TRAIN]: torch.utils.data.DataLoader(
                self._image_datasets[TRAIN],
                batch_size=BATCH_SIZE,
                shuffle=True,
            ),

            DATASETS[VALID]: torch.utils.data.DataLoader(
                self._image_datasets[VALID],
                batch_size=BATCH_SIZE,
                shuffle=False,
            ),

            DATASETS[TEST]: torch.utils.data.DataLoader(
                self._image_datasets[TEST],
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
        }

        self._class_names = self._image_datasets[TRAIN].classes
        self._num_classes = len(self._class_names)
        print(f"Data successfully loaded!")

    def get_batch_data(self, key=TRAIN) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function that returns a batch of images and labels
        :param key: the key for the dataloader to use
        :return: tuple of images and labels
        """
        # get the next iteration and return it
        image, labels = next(iter(self.dataloaders[key]))
        return image, labels

    def load_cat_to_name(self, filepath: str) -> None:
        """
        Function that loads the cat_to_name dictionary from file
        :param filepath: the filepath to the cat_to_name dictionary
        """
        with open(filepath, 'r') as f:
            self._cat_to_name = json.load(f)

    def get_class_to_idx(self, key: str =TRAIN) -> dict[str, int]:
        """
        Function that returns the mapping from class to index. Used when saving checkpoint
        :param key: key for the dataloader to use
        :return: class to index for dataloader
        """
        return self._image_datasets[key].class_to_idx

