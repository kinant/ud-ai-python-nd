import matplotlib.pyplot as plt
import numpy as np
from .utils import NORMALIZING_STD, NORMALIZING_MEAN
from PIL import Image
from math import floor
import torch

def plot_transformed_images(image_paths, transform):
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
def show_image(image, ax=None, title=None, normalize=True):
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

def plot_loss_curves(training_results):

    results = dict(list(training_results.items()))

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['valid_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['valid_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='valid_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='valid_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

# https://github.com/kinant/aipnd-project/blob/master/Image%20Classifier%20Project.ipynb
def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    # open the image
    image = Image.open(image_path)

    # print("Original Image size: ", image.size)

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


    # print("W: {}, H: {}".format(new_width, new_height))

    # resize the image
    image = image.resize((new_width, new_height))

    # print("Resized Image (keep aspect ratio): ", image.size)

    # perform center crop
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = image.size   # Get dimensions
    new_height, new_width = 224, 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))
    # print("cropped image size: ", image.size)

    # convert encoded color channels and convert to floats (divide by 255)
    np_image = np.array(image) / 255
    # print(np_image)

    # normalize
    np_image = (np_image - NORMALIZING_MEAN) / NORMALIZING_STD

    # finally, transpose
    # print("shape 1: ", np_image.shape)
    np_image = np_image.transpose((2, 0, 1))
    # print("transposed shape: ", np_image.shape)

    # Originally, I was returning a numpy array, as I thought these were the instructions, but
    # when trying to test, it would not work.
    # Found solution at: https://knowledge.udacity.com/questions/29173
    # We have to convert to a tensor before we return it
    return torch.Tensor(np_image)

def process_image_simple(image_path, transform):
    image = Image.open(image_path)
    image = transform(image)

    return image