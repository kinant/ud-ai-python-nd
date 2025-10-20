from typing import Any

import torch
from torch import nn

from PR02.helpers.torch_helpers import get_device
from torchvision import datasets, transforms, models
import PR02.helpers.utils as utils
from PR02.helpers.torch_helpers import get_device, print_device_info
from os import path
from collections import OrderedDict

import json

# IMAGE CONSTANTS
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
IMG_RESIZE = (255, 255)
BATCH_SIZE = 64
IMG_ROTATION = 45

# DATASET CONSTANTS
DATASETS = {
    'train': 'train',
    'valid': 'valid',
    'test': 'test',
}

# FOR MODEL DEFINITION AND TRAINING
ARCHITECTURES = {
    'vgg': 'vgg',
    'alexnet': 'alexnet',
    'densenet': 'densenet',
}

HIDDEN_UNITS = 1024
LEARNING_RATE = 0.003
DROP_OUT = 0.5
NUM_EPOCHS = 25
USE_CUDA = True

class ImageClassifier():

    def set_params_requires_grad_false(self):
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

    def init_model(self):
        if self._model_name == ARCHITECTURES['alexnet']:
            self._model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self._num_features = self._model.classifier[1].in_features

        elif self._model_name == ARCHITECTURES['densenet']:
            self._model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self._num_features = self._model.classifier.in_features

        else:
            self._model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._num_features = self._model.classifier[0].in_features

    def __init__(self, model_name=ARCHITECTURES['vgg'], n_hidden=HIDDEN_UNITS,
                 lr=LEARNING_RATE, n_epochs=NUM_EPOCHS, checkpoint_dir="", use_cuda=USE_CUDA):
        self._model_name = model_name
        self._model = None

        self._n_hidden = n_hidden
        self._lr = lr
        self._n_epochs = n_epochs
        self._checkpoint_dir = checkpoint_dir

        if use_cuda:
            self._device = get_device()
        else:
            self._device = get_device('cpu')

        self._data_dirs: dict = {}
        self._data_transforms: dict = {}
        self._image_datasets: dict = {}
        self._data_dict: dict = {}
        self._dataloaders: dict = {}
        self._dataset_sizes: dict = {}
        self.set_data_transforms()

        self._class_names: list = []
        self._cat_to_name: list = []
        self._num_classes = 0
        self._num_features = 0


    @property
    def model(self):
        return self._model

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
            DATASETS['train']: transforms.Compose([transforms.RandomRotation(IMG_ROTATION),
                                             transforms.RandomResizedCrop(IMG_SIZE),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(utils.NORMALIZING_MEAN, utils.NORMALIZING_STD)]),

            DATASETS['valid']: transforms.Compose([transforms.Resize(IMG_RESIZE),
                                             transforms.CenterCrop(IMG_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize(utils.NORMALIZING_MEAN, utils.NORMALIZING_STD)]),

            DATASETS['test']: transforms.Compose([transforms.Resize(IMG_RESIZE),
                                             transforms.CenterCrop(IMG_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize(utils.NORMALIZING_MEAN, utils.NORMALIZING_STD)])
        }

    def load_data(self, data_dir: str) -> None:

        self._data_dirs = {
            key: path.join(data_dir, DATASETS[key])
            for key in DATASETS.keys()
        }

        self._image_datasets = {
            key: datasets.ImageFolder(self._data_dirs[key], transform=self.data_transforms[key])
            for key in DATASETS.keys()
        }

        self._dataset_sizes = {key: len(self._image_datasets[key]) for key in DATASETS.keys()}

        self._dataloaders = {

            DATASETS['train']: torch.utils.data.DataLoader(
                self._image_datasets['train'],
                batch_size=BATCH_SIZE,
                shuffle=True,
            ),

            DATASETS['valid']: torch.utils.data.DataLoader(
                self._image_datasets['valid'],
                batch_size=BATCH_SIZE,
                shuffle=False,
            ),

            DATASETS['test']: torch.utils.data.DataLoader(
                self._image_datasets['test'],
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
        }

        self._class_names = self._image_datasets['train'].classes
        self._num_classes = len(self._class_names)

    def load_cat_to_name(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            self._cat_to_name = json.load(f)

    def show_model_summary(self) -> None:
        print(f'Model summary:')
        print(self._model)

    def set_classifier(self, n_hidden=HIDDEN_UNITS) -> None:
        self._model.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self._num_features, n_hidden)),
                ('drop', nn.Dropout(DROP_OUT)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(n_hidden, self._num_classes)),
                ('output', nn.LogSoftmax(dim=1))
            ]))

    def show_model_device_info(self):
        print_device_info(self._device)

    def get_class_to_idx(self):
        return self._image_datasets['train'].class_to_idx