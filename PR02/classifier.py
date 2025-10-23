# PROGRAMMER: Kinan Turman
# DATE UPDATED: Oct. 22, 2025
# PURPOSE: Defining a class for a custom image classifier

import torch
from torch import nn, optim
from torchvision import models

from helpers.torch_helpers import get_device, print_device_info
from os import path, makedirs
from collections import OrderedDict

from timeit import default_timer as timer
from copy import deepcopy

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


class ImageClassifier:

    @property
    def model(self) -> nn.Module:
        return self._model

    def __init__(self, model_name=ARCHITECTURES['alexnet'], n_classes=0, n_hidden=HIDDEN_UNITS,
                 lr=LEARNING_RATE, n_epochs=NUM_EPOCHS, checkpoint_dir="", use_cuda=USE_CUDA)-> None:
        """
        Initialize the ImageClassifier
        :param model_name: name of the pretrained model to use
        :param n_classes: number of classes in dataset (in other words, eventually the output size)
        :param n_hidden: number of hidden units in hidden layer
        :param lr: learning rate
        :param n_epochs: number of epochs for training
        :param checkpoint_dir: directory to save checkpoints
        :param use_cuda: flag to enable CUDA support
        """

        # Init and set class attributes
        self._model_name = model_name
        self._model = None

        self._n_hidden = n_hidden
        self._lr = lr
        self._n_epochs = n_epochs
        self._checkpoint_dir = checkpoint_dir

        # check if using CUDA or any other accelerator
        if use_cuda:
            self._device = get_device()
        else:
            # we can specifically use 'cpu' instead
            self._device = get_device('cpu')

        self._num_classes = n_classes
        self._num_features = 0

        self._criterion = None
        self._optimizer = None

        # Declare dictionary to store running training and validation results
        self._results = {
            "train_loss": 0,
            "train_acc": 0,
            "valid_loss": 0,
            "valid_acc": 0
        }

        # Declare dictionary to store all training and validation results
        self._results_dict = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": []
        }

        # Set the best state dict to None to start with
        self._best_state_dict = None

    def _set_params_requires_grad_false(self) -> None:
        """
        Function that sets model parameters to False. In other words it freezes them
        Since we are doing Transfer learning
        """
        if self._model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

    def init_model(self) -> None:
        """
        Function that initializes the model with a pre-trained model
        """

        # Determine which model to use based on the model_name
        # After checking which model, it loads the pretrained model with default weights
        # and sets the number of features based on the pre-trained model classifier in_features
        if self._model_name == ARCHITECTURES['alexnet']:
            self._model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self._num_features = self._model.classifier[1].in_features

        elif self._model_name == ARCHITECTURES['densenet']:
            self._model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self._num_features = self._model.classifier.in_features

        else:
            self._model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._num_features = self._model.classifier[0].in_features

        # Freeze parameters (for transfer learning)
        self._set_params_requires_grad_false()

    def show_model_summary(self) -> None:
        print(f'Model summary:')
        print(self._model)

    def set_classifier(self) -> None:
        """
        Function that sets the model's classifier
        We are using one hidden layer and one output layer
        """
        self._model.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self._num_features, self._n_hidden)),
                ('drop', nn.Dropout(DROP_OUT)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(self._n_hidden, self._num_classes)),
                ('output', nn.LogSoftmax(dim=1))
            ]))

    def show_device_info(self) -> None:
        """
        Function that prints information about the device used for training and inference
        """
        print_device_info(self._device)

    def _set_criterion_and_optimizer(self) -> None:
        """
        Function that sets the criterion and optimizer for use in training
        """
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._lr)

    def _train_step(self, dataloader: torch.utils.data.DataLoader) -> tuple[float, float]:
        """
        Function that performs one training step
        Reference used: https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
        :param dataloader: dataloader to use for training
        :return: train loss and train accuracy of the step (once per epoch)
        """

        # Put the model in train mode
        self._model.train()

        # Init train loss and train accuracy
        train_loss, train_correct = 0.0, 0.0

        # iterate over inputs and labels in the dataloader
        for inputs, labels in dataloader:

            # set the device
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

    def _valid_step(self, dataloader: torch.utils.data.DataLoader) -> tuple[float, float]:
        """
        Function that performs one validation step
        Reference used: https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
        :param dataloader: dataloader to use for validation
        :return: validation loss and validation accuracy of the step (once per epoch)
        """

        # Put model in eval mode
        self._model.eval()

        # Setup test loss and test accuracy values
        valid_loss, valid_correct = 0.0, 0.0

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

    def _train_results(self, epoch: int) -> float:
        """
        Function that prints out results one training step (one per epoch)
        and updates de results dictionary
        :param epoch: the current epoch
        :return the validation accuracy of the current epoch for use in determining the best weights
        """
        # Print Results
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {self._results["train_loss"]:.4f} | "
            f"train_acc: {self._results["train_acc"] * 100:.2f}% | "
            f"valid_loss: {self._results["valid_loss"]:.4f} | "
            f"valid_acc: {self._results["valid_acc"] * 100:.2f}%"
        )

        # Update Results dict by appending the new values
        self._results_dict["train_loss"].append(self._results["train_loss"])
        self._results_dict["train_acc"].append(self._results["train_acc"])
        self._results_dict["valid_loss"].append(self._results["valid_loss"])
        self._results_dict["valid_acc"].append(self._results["valid_acc"])

        return self._results["valid_acc"]

    def train(self, dataloaders: dict[str, torch.utils.data.DataLoader], num_epochs=NUM_EPOCHS) -> dict[str, float]:
        """
        Function that trains the model
        :param dataloaders: dataloaders to use for training and validation
        :param num_epochs: number of epochs to train for
        :return: dictionary of the results of the training
        """

        # set criterion and optimizer
        self._set_criterion_and_optimizer()

        # set the device
        self._model.to(self._device)

        # init results and results_dict
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

        # for the start time
        start_time = None

        # start of by making a copy of the initial state_dict (pre-trained)
        self._best_state_dict = deepcopy(self._model.state_dict())

        # to keep track of best accuracy
        best_acc = 0.0

        # I happened to use a library called tqdm to show a progress bar
        # One can install it if they want
        # Here we check if we have it installed and then use it
        try:
            from tqdm.auto import tqdm
            print("'tqdm' found")

            # get the start time
            start_time = timer()

            # iterate over each epoch and perform the training and validation steps
            for epoch in tqdm(range(num_epochs)):
                self._results["train_loss"], self._results["train_acc"] = self._train_step(dataloaders[TRAIN])
                self._results["valid_loss"], self._results["valid_acc"] = self._valid_step(dataloaders[VALID])

                # show the results and get the accuracy
                epoch_acc = self._train_results(epoch)

                # if we have better accuracy, then we set the new best_state_dict
                # and the new better accuracy
                if epoch_acc > best_acc:
                    self._best_state_dict = deepcopy(self._model.state_dict())
                    best_acc = epoch_acc

        # if tqdm not installed, default to not using it...
        # and do the same steps as above
        # this can be factored into a function, but for now I will leave it as is
        except ModuleNotFoundError:

            print("'tqdm' not found, using normal range function")

            start_time = timer()

            for epoch in range(num_epochs):
                self._results["train_loss"], self._results["train_acc"] = self._train_step(dataloaders[TRAIN])
                self._results["valid_loss"], self._results["valid_acc"] = self._valid_step(dataloaders[VALID])

                epoch_acc = self._train_results(epoch)

                if epoch_acc > best_acc:
                    self._best_state_dict = deepcopy(self._model.state_dict())
                    best_acc = epoch_acc

        # training complete, get the end time and show
        end_time = timer()
        elapsed_time = end_time - start_time
        print(f"Training Complete! Total training time: {elapsed_time} seconds")

        # Save best training weights
        self.model.load_state_dict(self._best_state_dict)

        # return the results dictionary (can be used for plotting)
        return self._results_dict

    def check_accuracy_on_test_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Function that checks accuracy on test data
        :param dataloader: the test dataloader
        """

        # To keep track of correct inferences and totals
        correct, total = 0, 0

        # set the device
        self._model.to(self._device)

        with torch.no_grad():
            # iterate over each batch of inputs and labels in the test dataloader
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                # get the probability logits
                logps = self._model(inputs)

                # get the top result
                top_p, top_class = logps.topk(1, dim=1)

                # get a tensor with all the top_class and label comparisons
                equals: torch.Tensor = torch.Tensor(top_class == labels.view(*top_class.shape))

                # we sum the above up to get update the number of corrects
                correct += equals.sum().item()

                # we also update the totals
                total += labels.size(0)

        # Once done, we print the accuracy of the network on test images
        print(f"Accuracy of the network on the test images:{(correct / total) * 100:.2f}%")

    def save_checkpoint(self, save_dir: str, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Function that saves the model
        :param save_dir: directory to save the checkpoint
        :param dataloader: dataloader to get the class_to_idx from
        :return:
        """

        # Get the checkpoint data
        checkpoint = {
            "model_arch": self._model_name,
            "output_size": self._num_classes,
            "hidden_units": self._n_hidden,
            "learning_rate": self._lr,
            "state_dict": self._model.state_dict(),
            "class_to_idx": dataloader.get_class_to_idx()
        }

        # Create the filepath
        filepath = path.join(save_dir, "checkpoint.pth")

        print(f"Saving checkpoint to: {filepath}")

        # If the directory does not exist, we create it
        if not path.exists(save_dir):
            makedirs(save_dir)

        # save
        torch.save(checkpoint, filepath)

def load_checkpoint(file_path: str) -> ImageClassifier:
    """
    Function to load a checkpoint
    :param filepath: filepath to load the checkpoint from
    :return: ImageClassifier as loaded with checkpoint
    """

    print(f"Loading checkpoint from {file_path}")

    # Load the checkpoint
    checkpoint = torch.load(file_path)

    # Get data from checkpoint dictionary
    model_arch = checkpoint["model_arch"]
    n_hidden = checkpoint["hidden_units"]
    learning_rate = checkpoint["learning_rate"]
    class_to_idx = checkpoint["class_to_idx"]

    # Init a new ImageClassifier
    new_classifier = ImageClassifier(model_arch, n_hidden=n_hidden, lr=learning_rate)
    new_classifier._num_classes = checkpoint["output_size"]

    # Init model and set the classifier
    new_classifier.init_model()
    new_classifier.set_classifier()

    # Load the state dict, and class_to_idx into the model
    new_classifier.model.load_state_dict(checkpoint["state_dict"])
    new_classifier.model.class_to_idx = class_to_idx
    print(f"Loading checkpoint...COMPLETE")

    return new_classifier