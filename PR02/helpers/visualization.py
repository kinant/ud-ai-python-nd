# PROGRAMMER: Kinan Turman
# DATE UPDATED: Oct. 22, 2025
# PURPOSE: Helper functions for visualization and processing images

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2

from .utils import NORMALIZING_STD, NORMALIZING_MEAN
from PIL import Image
from math import floor
import torch

def plot_transformed_images(
        image_paths: list[str],
        transform: torchvision.transforms.v2.Transform) -> None:
    """
    Function to plot original and transformed images
    Used as reference: https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
    :param image_paths: list of image paths to plot
    :param transform: the transformation applied to each image
    """
    for image_path in image_paths:
        with Image.open(image_path) as img:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img)
            ax[0].set_title(f"Original \nSize: {img.size}")
            ax[0].axis("off")

            # Transform and plot image
            transformed_image = transform(img).numpy().transpose((1, 2, 0))

            # Normalize
            transformed_image = (NORMALIZING_STD * transformed_image) + NORMALIZING_MEAN
            transformed_image = np.clip(transformed_image, 0, 1)

            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def show_image(
        image:torch.Tensor,
        ax=None,
        title: str=None,
        normalize:bool=True) -> None:
    """
    Function to show an image
    :param image: image to show
    :param ax: axis
    :param title: title to display
    :param normalize: flag to normalize the image
    """


    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    ax.set_title(title)

    return ax

def plot_loss_curves(results: dict) -> None:
    """
    Function to plot loss curves
    Used as reference: https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
    :param results: training results dictionary
    """

    # Get the loss values
    train_loss = results['train_loss']
    valid_loss = results['valid_loss']

    # Get the accuracy values
    train_accuracy = results['train_acc']
    valid_accuracy = results['valid_acc']

    # Determine how many epochs
    epochs = range(len(results['train_loss']))

    # Setup plot
    plt.figure(figsize=(15, 7))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, valid_loss, label='valid_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='train_accuracy')
    plt.plot(epochs, valid_accuracy, label='valid_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

# https://github.com/kinant/aipnd-project/blob/master/Image%20Classifier%20Project.ipynb
def process_image(image_path: str) -> torch.Tensor:
    """
    Function to process image into a torch tensor
    Used as reference:
        https://github.com/kinant/aipnd-project/blob/master/Image%20Classifier%20Project.ipynb
        This is actually my project submission from years ago

    :param image_path: path of the image
    :return: the image as a tensor
    """

    # open the image
    image = Image.open(image_path)

    # first resize the images where the shortest side is 256 px
    width, height = image.size
    size = 256, 256

    new_width, new_height = None, None

    # if the height is the shorter side
    if height < width:
        # find ratio between larger and smaller side
        ratio = float(width) / float(height)
        # resize smaller side to 256
        new_height = 256
        # resize larger side to 256 * ratio
        new_width = int(floor(ratio * size[0]))
    # else, the width is the shorter side
    else:
        # find ratio between larger and smaller side
        ratio = float(height)/float(width)
        # resize smaller side to 256
        new_width = 256
        # resize larger side to 256 * ratio
        new_height = int(floor(ratio * size[1]))

    # resize the image
    image = image.resize((new_width, new_height))

    # perform center crop
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = image.size   # Get dimensions
    new_height, new_width = 224, 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))

    # convert encoded color channels and convert to floats (divide by 255)
    np_image = np.array(image) / 255

    # normalize
    np_image = (np_image - NORMALIZING_MEAN) / NORMALIZING_STD

    # finally, transpose
    np_image = np_image.transpose((2, 0, 1))
    # print("transposed shape: ", np_image.shape)

    # Originally, I was returning a numpy array, as I thought these were the instructions, but
    # when trying to test, it would not work.
    # Found solution at: https://knowledge.udacity.com/questions/29173
    # We have to convert to a tensor before we return it
    return torch.Tensor(np_image)

def process_image_simple(image_path: str,
                         transform: torchvision.transforms.v2.Transform) -> torch.Tensor:
    """
    Simpler function to process image into a torch tensor
    :param image_path: image path
    :param transform: transformation to apply to image
    :return: processed image as a torch tensor
    """

    # Open image and apply transform
    image = Image.open(image_path)
    image = transform(image)

    return image