from typing import Any

import torch
from torch import nn

from PR02.helpers.torch_helpers import get_device
from torchvision import datasets, transforms, models
import PR02.helpers.utils as utils
from os import path

import json

class ImageClassifier():

    # IMAGE CONSTANTS
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
    IMG_RESIZE = (255, 255)
    BATCH_SIZE = 64
    IMG_ROTATION = 45

    # DATASET CONSTANTS
    DATASETS = {
        'train':'train',
        'valid':'valid',
        'test':'test',
    }

    # FOR MODEL DEFINITION AND TRAINING
    ARCHITECTURES = {
        'vgg': 'vgg',
        'alexnet': 'alexnet',
        'densenet': 'densenet',
    }

    def set_params_requires_grad(self):
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

    def init_model(self, model_name='vgg'):
        model = None

        if model_name == self.ARCHITECTURES['alexnet']:
            model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

        return model

    def __init__(self, model='vgg'):
        self._model = self.init_model(model)
        self._device = get_device()

        self._data_dirs: dict = {}
        self._data_transforms: dict = {}
        self._image_datasets: dict = {}
        self._data_dict: dict = {}
        self._dataloaders: dict = {}
        self._dataset_sizes: dict = {}

        self._class_names: list = []
        self._cat_to_name: list = []

        self.set_data_transforms()

    @property
    def data_transforms(self) -> dict:
        return self._data_transforms

    @property
    def data_dirs(self) -> dict:
        return self._data_dirs

    @property
    def image_datasets(self) -> dict:
        return self._image_datasets

    @property
    def dataloaders(self) -> dict:
        return self._dataloaders

    @property
    def dataset_sizes(self) -> dict:
        return self._dataset_sizes

    @property
    def cat_to_name(self) -> list:
        return self._cat_to_name

    def set_data_transforms(self) -> None:
        self._data_transforms = {
            self.DATASETS['train']: transforms.Compose([transforms.RandomRotation(self.IMG_ROTATION),
                                             transforms.RandomResizedCrop(self.IMG_SIZE),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(utils.NORMALIZING_MEAN, utils.NORMALIZING_STD)]),

            self.DATASETS['valid']: transforms.Compose([transforms.Resize(self.IMG_RESIZE),
                                             transforms.CenterCrop(self.IMG_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize(utils.NORMALIZING_MEAN, utils.NORMALIZING_STD)]),

            self.DATASETS['test']: transforms.Compose([transforms.Resize(self.IMG_RESIZE),
                                             transforms.CenterCrop(self.IMG_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize(utils.NORMALIZING_MEAN, utils.NORMALIZING_STD)])
        }

    def load_data(self, data_dir: str) -> None:

        self._data_dirs = {
            key: path.join(data_dir, self.DATASETS[key])
            for key in self.DATASETS.keys()
        }

        self._image_datasets = {
            key: datasets.ImageFolder(self._data_dirs[key], transform=self.data_transforms[key])
            for key in self.DATASETS.keys()
        }

        self._dataset_sizes = {key: len(self._image_datasets[key]) for key in self.DATASETS.keys()}

        self._dataloaders = {

            self.DATASETS['train']: torch.utils.data.DataLoader(
                self._image_datasets['train'],
                batch_size=self.BATCH_SIZE,
                shuffle=True,
            ),

            self.DATASETS['valid']: torch.utils.data.DataLoader(
                self._image_datasets['valid'],
                batch_size=self.BATCH_SIZE,
                shuffle=False,
            ),

            self.DATASETS['test']: torch.utils.data.DataLoader(
                self._image_datasets['test'],
                batch_size=self.BATCH_SIZE,
                shuffle=False,
            )
        }

        self._class_names = self._image_datasets['train'].classes

    def load_cat_to_name(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            self._cat_to_name = json.load(f)
