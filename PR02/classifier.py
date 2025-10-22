import copy
from typing import Any

import torch
from torch import nn, optim

from torchvision import datasets, transforms, models
import helpers.utils as utils
from helpers.torch_helpers import get_device, print_device_info
from os import path, makedirs
from collections import OrderedDict

import json
from timeit import default_timer as timer

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

# PHASE/DATALOADERS CONSTANTS
TRAIN = "train"
TEST = "test"
VALID = "valid"

def load_checkpoint(filepath):
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath)

    model_arch = checkpoint["model_arch"]
    n_hidden = checkpoint["hidden_units"]
    learning_rate = checkpoint["learning_rate"]
    class_to_idx = checkpoint["class_to_idx"]

    new_classifier = ImageClassifier(model_arch, n_hidden=n_hidden, lr=learning_rate)
    new_classifier._num_classes = checkpoint["output_size"]

    new_classifier.init_model()
    new_classifier.set_classifier()

    new_classifier.model.load_state_dict(checkpoint["state_dict"])
    new_classifier.model.class_to_idx = class_to_idx
    print(f"Loading checkpoint from {filepath} COMPLETE")

    return new_classifier

class ImageClassifier:

    def set_params_requires_grad_false(self):
        if self._model is not None:
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

        self.set_params_requires_grad_false()

    def __init__(self, model_name=ARCHITECTURES['vgg'], n_classes=0, n_hidden=HIDDEN_UNITS,
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

        self._num_classes = n_classes
        self._num_features = 0

        self._criterion = None
        self._optimizer = None

        self._results = {
            "train_loss": 0,
            "train_acc": 0,
            "valid_loss": 0,
            "valid_acc": 0
        }

        self._results_dict = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": []
        }

        self._best_state_dict = None

    @property
    def model(self):
        return self._model

    def show_model_summary(self) -> None:
        print(f'Model summary:')
        print(self._model)

    def set_classifier(self) -> None:
        self._model.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self._num_features, self._n_hidden)),
                ('drop', nn.Dropout(DROP_OUT)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(self._n_hidden, self._num_classes)),
                ('output', nn.LogSoftmax(dim=1))
            ]))

    def show_model_device_info(self):
        print_device_info(self._device)

    def set_criterion_and_optimizer(self):
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=LEARNING_RATE)

    # https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
    def train_step(self, dataloader):
        # Put the model in train mode
        self._model.train()

        # Init train loss and train accuracy
        train_loss, train_correct = 0, 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self._device), labels.to(self._device)

            # Reset gradients
            self._optimizer.zero_grad()

            # Forward pass
            logps = self._model(inputs)
            loss = self._criterion(logps, labels)

            # Add to the training loss
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update weights
            self._optimizer.step()

            # Get the probabilities
            ps = torch.exp(logps)

            # Get the top class
            _, top_class = ps.topk(1, dim=1)

            # Get list of all equalities top_class == label
            equals: torch.Tensor = torch.Tensor(top_class == labels.view(*top_class.shape))

            # print(f"TRAIN EQUALS: {equals}")

            # Update training corrects
            train_correct += torch.sum(equals).item()
            # print(f"Running Train Corrects: {train_correct}")

        # Calculate metrics
        train_loss = train_loss / len(dataloader.dataset)
        train_accuracy = train_correct / len(dataloader.dataset)

        return train_loss, train_accuracy

    def valid_step(self, dataloader):
        # Put model in eval mode
        self._model.eval()

        # Setup test loss and test accuracy values
        valid_loss, valid_correct = 0, 0

        # Turn on inference context manager
        with torch.no_grad():
            # Loop through DataLoader batches
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                # 1. Forward pass
                logps = self._model(inputs)

                # 2. Calculate and accumulate loss
                loss = self._criterion(logps, labels)
                valid_loss += loss.item()

                # Calculate and accumulate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals: torch.Tensor = torch.Tensor(top_class == labels.view(*top_class.shape))

                # print(f"VALID EQUALS: {equals}")

                valid_correct += torch.sum(equals).item()
                # print(f"Running Valid Corrects: {valid_correct}")

        # Adjust metrics to get average loss and accuracy per batch
        valid_loss = valid_loss / len(dataloader.dataset)
        valid_acc = valid_correct / len(dataloader.dataset)
        return valid_loss, valid_acc

    def train_results(self, epoch):

        # Print Results
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {self._results["train_loss"]:.4f} | "
            f"train_acc: {self._results["train_acc"] * 100:.2f}% | "
            f"valid_loss: {self._results["valid_loss"]:.4f} | "
            f"valid_acc: {self._results["valid_acc"] * 100:.2f}%"
        )

        # Update Results dict
        self._results_dict["train_loss"].append(self._results["train_loss"])
        self._results_dict["train_acc"].append(self._results["train_acc"])
        self._results_dict["valid_loss"].append(self._results["valid_loss"])
        self._results_dict["valid_acc"].append(self._results["valid_acc"])

        return self._results["valid_acc"]

    def train(self, dataloaders, num_epochs=NUM_EPOCHS):

        self.set_criterion_and_optimizer()
        self._model.to(self._device)

        self._results = {
            "train_loss": 0,
            "train_acc": 0,
            "valid_loss": 0,
            "valid_acc": 0
        }

        self._results_dict = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": []
        }

        start_time = None

        self._best_state_dict = copy.deepcopy(self._model.state_dict())
        best_acc = 0.0

        try:
            from tqdm.auto import tqdm
            print("'tqdm' found")

            start_time = timer()

            for epoch in tqdm(range(num_epochs)):
                self._results["train_loss"], self._results["train_acc"] = self.train_step(dataloaders[TRAIN])
                self._results["valid_loss"], self._results["valid_acc"] = self.valid_step(dataloaders[VALID])

                epoch_acc = self.train_results(epoch)

                if epoch_acc > best_acc:
                    self._best_state_dict = copy.deepcopy(self._model.state_dict())
                    best_acc = epoch_acc

        except ModuleNotFoundError:
            print("'tqdm' not found, using normal range function")

            start_time = timer()

            for epoch in range(num_epochs):
                self._results["train_loss"], self._results["train_acc"] = self.train_step(dataloaders[TRAIN])
                self._results["valid_loss"], self._results["valid_acc"] = self.valid_step(dataloaders[VALID])

                epoch_acc = self.train_results(epoch)

                if epoch_acc > best_acc:
                    self._best_state_dict = copy.deepcopy(self._model.state_dict())
                    best_acc = epoch_acc

        end_time = timer()
        elapsed_time = end_time - start_time
        print(f"Training Complete! Total training time: {elapsed_time} seconds")

        # Save best training weights
        self.model.load_state_dict(self._best_state_dict)

        return self._results_dict

    def check_accuracy_on_test_data(self, dataloader):
        correct, total = 0, 0

        self._model.to(self._device)

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                logps = self._model(inputs)
                top_p, top_class = logps.topk(1, dim=1)
                equals: torch.Tensor = torch.Tensor(top_class == labels.view(*top_class.shape))
                correct += equals.sum().item()
                total += labels.size(0)

        print(f"Accuracy of the network on the test images:{(correct / total) * 100:.2f}%")

    def save_checkpoint(self, save_dir, dataloader):

        checkpoint = {
            "model_arch": "vgg",
            "output_size": self._num_classes,
            "hidden_units": self._n_hidden,
            "learning_rate": self._lr,
            "state_dict": self._model.state_dict(),
            "class_to_idx": dataloader.get_class_to_idx()
        }

        filepath = path.join("checkpoints/", "checkpoint.pth")

        if not path.exists("checkpoints/"):
            makedirs("checkpoints/")

        torch.save(checkpoint, filepath)
